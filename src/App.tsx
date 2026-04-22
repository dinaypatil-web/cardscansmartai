/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, RefreshCw, UserPlus, Check, AlertCircle, Loader2, ScanLine, X, Globe, FileText, Mail, MapPin, Phone, Smartphone, ImagePlus, Zap, Cpu } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI, Type } from "@google/genai";
import confetti from 'canvas-confetti';
import { createWorker } from 'tesseract.js';
import { parseContactFromText } from './contactParser';

// --- Types ---

interface ContactInfo {
  firstName: string;
  lastName: string;
  title?: string;
  company?: string;
  landlines: string[];
  mobiles: string[];
  email?: string;
  website?: string;
  address?: string;
  notes?: string;
}

// --- Utilities ---

const generateVCard = (contact: ContactInfo): string => {
  const fullName = `${contact.firstName} ${contact.lastName}`.trim();
  const vcard = [
    'BEGIN:VCARD',
    'VERSION:3.0',
    `FN:${fullName}`,
    `N:${contact.lastName};${contact.firstName};;;`,
    contact.company ? `ORG:${contact.company}` : '',
    contact.title ? `TITLE:${contact.title}` : '',
    ...contact.landlines.map(num => `TEL;TYPE=WORK,VOICE:${num}`),
    ...contact.mobiles.map(num => `TEL;TYPE=CELL,VOICE:${num}`),
    contact.email ? `EMAIL;TYPE=INTERNET:${contact.email}` : '',
    contact.website ? `URL:${contact.website}` : '',
    contact.address ? `ADR;TYPE=WORK:;;${contact.address.replace(/\n/g, ';')}` : '',
    `NOTE:${contact.notes ? contact.notes + ' | ' : ''}Scanned with CardScan AI on ${new Date().toLocaleDateString()}`,
    'END:VCARD'
  ].filter(Boolean).join('\n');

  return vcard;
};

const downloadVCard = (contact: ContactInfo) => {
  const vcard = generateVCard(contact);
  const fullName = `${contact.firstName} ${contact.lastName}`.trim();
  const blob = new Blob([vcard], { type: 'text/vcard' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${fullName.replace(/\s+/g, '_')}.vcf`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Resize image for AI vision APIs.
 * 1600px / 0.9 quality — preserves legibility of 8pt fonts on business cards.
 */
const resizeImage = (base64Str: string, maxWidth = 1600, maxHeight = 1600): Promise<string> => {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = base64Str;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      let width = img.width;
      let height = img.height;

      if (width > height) {
        if (width > maxWidth) { height = Math.round(height * maxWidth / width); width = maxWidth; }
      } else {
        if (height > maxHeight) { width = Math.round(width * maxHeight / height); height = maxHeight; }
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0, width, height);
      }
      resolve(canvas.toDataURL('image/jpeg', 0.9));
    };
  });
};

/**
 * Preprocess image for Tesseract.js OCR.
 * Pipeline: upscale → grayscale → Gaussian denoise → unsharp mask (sharpen) → Otsu auto-threshold.
 */
const preprocessImageForOCR = (base64Str: string, maxWidth = 2400, maxHeight = 2400): Promise<string> => {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = base64Str;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      let width = img.width;
      let height = img.height;

      // Upscale small images — Tesseract reads best at ≥300 DPI equivalent
      const scale = Math.min(3, Math.max(1, 2400 / Math.max(width, height)));
      width = Math.round(width * scale);
      height = Math.round(height * scale);
      if (width > maxWidth) { height = Math.round(height * maxWidth / width); width = maxWidth; }
      if (height > maxHeight) { width = Math.round(width * maxHeight / height); height = maxHeight; }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d')!;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, width, height);

      const imageData = ctx.getImageData(0, 0, width, height);
      const data = imageData.data;
      const total = width * height;

      // Step 1: Weighted grayscale (ITU-R BT.601)
      const gray = new Float32Array(total);
      for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      }

      // Step 2: 3×3 Gaussian blur for noise reduction
      const blurred = new Float32Array(total);
      const k = [1/16, 2/16, 1/16, 2/16, 4/16, 2/16, 1/16, 2/16, 1/16];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let acc = 0;
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const ny = Math.min(height - 1, Math.max(0, y + ky));
              const nx = Math.min(width - 1, Math.max(0, x + kx));
              acc += gray[ny * width + nx] * k[(ky + 1) * 3 + (kx + 1)];
            }
          }
          blurred[y * width + x] = acc;
        }
      }

      // Step 3: Unsharp mask — amplifies edges so Tesseract reads fine fonts reliably
      const sharpened = new Float32Array(total);
      for (let p = 0; p < total; p++) {
        sharpened[p] = Math.min(255, Math.max(0, gray[p] + 1.5 * (gray[p] - blurred[p])));
      }

      // Step 4: Otsu's method — automatically finds the optimal binarization threshold
      const hist = new Uint32Array(256);
      for (let p = 0; p < total; p++) hist[Math.round(sharpened[p])]++;
      let sumAll = 0;
      for (let t = 0; t < 256; t++) sumAll += t * hist[t];
      let sumB = 0, wB = 0, maxVar = 0, otsu = 128;
      for (let t = 0; t < 256; t++) {
        wB += hist[t]; if (!wB) continue;
        const wF = total - wB; if (!wF) break;
        sumB += t * hist[t];
        const mB = sumB / wB, mF = (sumAll - sumB) / wF;
        const varBetween = wB * wF * (mB - mF) ** 2;
        if (varBetween > maxVar) { maxVar = varBetween; otsu = t; }
      }

      // Step 5: Apply threshold — white bg, black text (ideal for Tesseract)
      for (let p = 0, i = 0; p < total; p++, i += 4) {
        const val = sharpened[p] > otsu ? 255 : 0;
        data[i] = data[i + 1] = data[i + 2] = val;
        data[i + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
  });
};
// --- Components ---

export default function App() {
  const [images, setImages] = useState<string[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [contact, setContact] = useState<ContactInfo | null>(null);
  const [partialContact, setPartialContact] = useState<Partial<ContactInfo> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cooldown, setCooldown] = useState<number>(0);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [step, setStep] = useState<'front' | 'back' | 'review' | 'analyzing' | 'result'>('front');
  const [showFlash, setShowFlash] = useState(false);
  const [activeProvider, setActiveProvider] = useState<'gemini' | 'groq' | 'openrouter' | 'tesseract' | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileInputBackRef = useRef<HTMLInputElement>(null);

  // Initialize Camera
  useEffect(() => {
    if ((step === 'front' || step === 'back') && !contact) {
      startCamera();
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [step, contact]);

  // Cooldown timer
  useEffect(() => {
    if (cooldown > 0) {
      const timer = setTimeout(() => setCooldown(cooldown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [cooldown]);

  const startCamera = async () => {
    setError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Your browser does not support camera access.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraReady(true);
        setError(null);
      }
    } catch (err: any) {
      console.error("Error accessing camera:", err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError("Camera access was denied. Please enable it in your browser settings and click 'Retry'.");
      } else {
        setError("Could not access camera. Please ensure you are using a secure connection (HTTPS) and have granted permissions.");
      }
      setIsCameraReady(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
  };

  const captureImage = useCallback(() => {
    if (videoRef.current && canvasRef.current && !showFlash) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg');

        // Flash & Sound effect
        setShowFlash(true);
        setTimeout(() => setShowFlash(false), 300);

        const newImages = [...images, dataUrl];
        setImages(newImages);

        // Brief delay to "see" the captured image before moving to next step
        setTimeout(() => {
          if (step === 'front') {
            setStep('back');
          } else {
            setStep('review');
          }
        }, 500);
      }
    }
  }, [images, step, showFlash]);

  const skipBackSide = () => {
    setStep('review');
  };

  const handleGalleryUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>, side: 'front' | 'back') => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Accept only image types
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      if (!dataUrl) return;

      // Flash effect for visual feedback
      setShowFlash(true);
      setTimeout(() => setShowFlash(false), 300);

      const newImages = side === 'front' ? [dataUrl] : [...images, dataUrl];
      setImages(newImages);

      setTimeout(() => {
        if (side === 'front') {
          setStep('back');
        } else {
          setStep('review');
        }
      }, 400);
    };
    reader.readAsDataURL(file);

    // Reset file input so same file can be re-selected
    e.target.value = '';
  }, [images]);

  // ─── Shared prompt & helpers ──────────────────────────────────────────────────
  const CARD_PROMPT = `You are a world-class business card OCR specialist. Extract ALL contact information with maximum accuracy.

CRITICAL READING RULES:
1. Read EVERY visible character — including 6-8pt text, faint ink, embossed/foil print, and stylized fonts.
2. Recognize ALL languages: English, Hindi, Marathi, Gujarati, Tamil, Telugu, Kannada, Bengali, etc. For non-Latin scripts, provide phonetic transliteration.
3. Multiple images = FRONT and BACK of ONE card — merge ALL data intelligently, never duplicate.
4. Correct known OCR confusions you spot: I↔1, O↔0, l↔|, S↔5, B↔8, G↔6.

FIELD-BY-FIELD EXTRACTION GUIDE:

firstName + lastName — The HUMAN person's name only.
  • Honorifics hint at a name line: Mr., Mrs., Ms., Dr., Prof., Shri, Smt., Er., Adv., CA, CS, Ar.
  • ALL-CAPS names are very common: "RAJESH KUMAR SHARMA" → firstName:"Rajesh", lastName:"Kumar Sharma".
  • Do NOT place company/brand name here.

company — Full registered business name.
  • Entity suffixes: Pvt. Ltd., Ltd., LLP, OPC, Proprietorship, Partnership.
  • Business words: Enterprises, Associates, Industries, Trading, Services, Solutions, Technologies, Consultants, Builders, Infra, Group.

title — Exact job designation.
  • Common Indian titles: Managing Director, MD, Director, Proprietor, Partner, Founder, CEO, GM, DGM, AGM, VP, Manager, Executive, Officer, Engineer, Architect, Consultant, Advocate, CA, CS.
  • Preserve full phrasing: "Senior Sales Executive" not just "Executive".

mobiles — Mobile/cellular numbers ONLY.
  • Indian mobiles: 10 digits starting with 6, 7, 8, or 9.
  • With country code: +91 98765 43210 or +91-9876543210.
  • Labels: M:, Mob:, Cell:, Mobile:, WhatsApp:, WA:.
  • Capture ALL mobile numbers — some cards have 2-3.

landlines — Office, landline, and fax numbers ONLY.
  • Indian STD format: 2-4 digit city code + 6-8 digit number.
  • City codes: 022 (Mumbai), 011 (Delhi), 080 (Bangalore), 044 (Chennai), 033 (Kolkata), 040 (Hyderabad), 020 (Pune), 079 (Ahmedabad), 0261 (Surat), 0712 (Nagpur).
  • Examples: (022) 2345-6789, 011-23456789, 0261-2345678.
  • Labels: T:, Tel:, Ph:, Phone:, Off:, Office:, Fax:, F:, Res:.
  • Fax numbers go here.

email — Email address containing @. Extract verbatim.

website — Website URL. Add "https://" prefix if missing.

address — Complete postal address as one string.
  • Include: shop/flat/gala/plot number, building, floor, street/road, area, city, state, PIN code.
  • Indian terms: Gala No., Shop No., Plot No., S.No., Opp., Near, S.V. Road, Nagar, Colony, Chowk, Industrial Estate, MIDC, GIDC.
  • Join multi-line fragments with ", ".

notes — All remaining text not fitting above:
  • GSTIN: 15-char format starting with 2-digit state code (e.g. 27AABCU9603R1ZX).
  • PAN: 10-char AAAAA9999A format.
  • Udyam/MSME: starts with "UDYAM-" (e.g. UDYAM-MH-04-0123456).
  • CIN: starts with L or U followed by numbers.
  • Social media: @handle, LinkedIn, Twitter/X, Instagram.
  • ISO certifications, taglines, slogans, QR labels, service lists, any other identifiers.

ZERO-HALLUCINATION RULES:
• If a field is absent/illegible → return "" or []. NEVER guess or invent data.
• Mobile and landline are MUTUALLY EXCLUSIVE — each number goes in exactly ONE array, never both.
• Do NOT duplicate any number across arrays.
• Preserve original number formatting (spaces, dashes, country codes).

Return ONLY this exact JSON (no markdown, no explanation, nothing else):
{"firstName":"","lastName":"","title":"","company":"","landlines":[],"mobiles":[],"email":"","website":"","address":"","notes":""}`;

  const sanitizeResult = (r: any): ContactInfo => ({
    firstName: r.firstName || '',
    lastName: r.lastName || '',
    title: r.title || '',
    company: r.company || '',
    email: r.email || '',
    website: r.website || '',
    address: r.address || '',
    notes: r.notes || '',
    landlines: Array.isArray(r.landlines) ? r.landlines.filter(Boolean) : [],
    mobiles: Array.isArray(r.mobiles) ? r.mobiles.filter(Boolean) : [],
  });

  const hasUsefulInfo = (r: any) =>
    r.firstName || r.lastName || r.company || r.email ||
    (r.mobiles?.length > 0) || (r.landlines?.length > 0);

  // ─── Tier 0: Gemini (streaming, partial preview) ────────────────────────────
  const scanWithGemini = async (base64Images: string[]): Promise<ContactInfo> => {
    // @ts-ignore
    const apiKey = (import.meta as any).env?.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === 'MY_GEMINI_API_KEY' || apiKey.includes('YOUR_API_KEY')) {
      throw new Error('NO_KEY');
    }
    const ai = new GoogleGenAI({ apiKey });
    const compressed = await Promise.all(base64Images.map(img => resizeImage(img)));
    const imageParts = compressed.map(img => ({
      inlineData: { mimeType: 'image/jpeg' as const, data: img.split(',')[1] }
    }));
    const stream = await ai.models.generateContentStream({
      model: 'gemini-2.0-flash',
      contents: [{ role: 'user', parts: [{ text: CARD_PROMPT }, ...imageParts] }],
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            firstName: { type: Type.STRING }, lastName: { type: Type.STRING },
            title: { type: Type.STRING }, company: { type: Type.STRING },
            landlines: { type: Type.ARRAY, items: { type: Type.STRING } },
            mobiles: { type: Type.ARRAY, items: { type: Type.STRING } },
            email: { type: Type.STRING }, website: { type: Type.STRING },
            address: { type: Type.STRING }, notes: { type: Type.STRING }
          }
        }
      }
    });
    let fullText = '';
    // @ts-ignore
    for await (const chunk of stream.stream) {
      fullText += chunk.text || '';
      try {
        const cleaned = fullText.trim();
        if (cleaned.startsWith('{')) {
          const result: any = {};
          ['firstName','lastName','title','company','email','website','address','notes'].forEach(f => {
            const m = [...cleaned.matchAll(new RegExp(`"${f}"\\s*:\\s*"([^"]*)"`, 'g'))].pop();
            if (m) result[f] = m[1];
          });
          ['landlines','mobiles'].forEach(f => {
            const m = [...cleaned.matchAll(new RegExp(`"${f}"\\s*:\\s*\\[([^\\]]*)\\]`, 'g'))].pop();
            if (m) { try { result[f] = m[1].trim() ? m[1].split(',').map((s: string) => s.trim().replace(/^"|"$|^'|'$/g,'')).filter(Boolean) : []; } catch { /**/ } }
          });
          if (Object.keys(result).length > 0) setPartialContact(prev => ({ ...prev, ...result }));
        }
      } catch { /**/ }
    }
    const r = JSON.parse(fullText || '{}');
    if (!hasUsefulInfo(r)) throw new Error('NO_INFO');
    return sanitizeResult(r);
  };

  // ─── Tier 1: Groq Vision (free, OpenAI-compatible) ─────────────────────────
  const scanWithGroq = async (base64Images: string[]): Promise<ContactInfo> => {
    // @ts-ignore
    const apiKey = (import.meta as any).env?.VITE_GROQ_API_KEY;
    if (!apiKey || apiKey.includes('YOUR')) throw new Error('NO_KEY');
    const compressed = await Promise.all(base64Images.map(img => resizeImage(img)));
    const imageContent = compressed.map(img => ({ type: 'image_url', image_url: { url: img } }));
    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'meta-llama/llama-4-scout-17b-16e-instruct',
        messages: [{ role: 'user', content: [{ type: 'text', text: CARD_PROMPT }, ...imageContent] }],
        response_format: { type: 'json_object' },
        max_tokens: 1024,
        temperature: 0.1,
      })
    });
    if (!res.ok) {
      const err = await res.text();
      if (res.status === 429) throw new Error('QUOTA');
      throw new Error(`Groq error ${res.status}: ${err}`);
    }
    const json = await res.json();
    const r = JSON.parse(json.choices[0]?.message?.content || '{}');
    if (!hasUsefulInfo(r)) throw new Error('NO_INFO');
    setPartialContact(r);
    return sanitizeResult(r);
  };

  // ─── Tier 2: OpenRouter free vision model ──────────────────────────────────
  const scanWithOpenRouter = async (base64Images: string[]): Promise<ContactInfo> => {
    // @ts-ignore
    const apiKey = (import.meta as any).env?.VITE_OPENROUTER_API_KEY;
    if (!apiKey || apiKey.includes('YOUR')) throw new Error('NO_KEY');
    const compressed = await Promise.all(base64Images.map(img => resizeImage(img)));
    const imageContent = compressed.map(img => ({ type: 'image_url', image_url: { url: img } }));
    const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': window.location.origin,
        'X-Title': 'CardScan AI',
      },
      body: JSON.stringify({
        model: 'meta-llama/llama-3.2-11b-vision-instruct:free',
        messages: [{ role: 'user', content: [{ type: 'text', text: CARD_PROMPT }, ...imageContent] }],
        response_format: { type: 'json_object' },
        max_tokens: 1024,
        temperature: 0.1,
      })
    });
    if (!res.ok) {
      const err = await res.text();
      if (res.status === 429) throw new Error('QUOTA');
      throw new Error(`OpenRouter error ${res.status}: ${err}`);
    }
    const json = await res.json();
    const r = JSON.parse(json.choices[0]?.message?.content || '{}');
    if (!hasUsefulInfo(r)) throw new Error('NO_INFO');
    setPartialContact(r);
    return sanitizeResult(r);
  };

  // ─── Tier 3: Tesseract.js — offline, no API needed ──────────────────────────
  const scanWithTesseract = async (base64Images: string[]): Promise<ContactInfo> => {
    // OEM 1 = LSTM neural net (most accurate). PSM 6 = uniform text block.
    const worker = await createWorker('eng', 1, { logger: () => {} });
    await worker.setParameters({
      tessedit_pageseg_mode: '6' as any,    // PSM 6: single uniform block
      tessedit_ocr_engine_mode: '1' as any, // OEM 1: LSTM only
      preserve_interword_spaces: '1' as any,
    });

    try {
      let combinedText = '';

      for (const img of base64Images) {
        const preprocessed = await preprocessImageForOCR(img);

        // Pass 1 — PSM 6: treats card as a neat text block (handles most cards well)
        const { data: { text: text6 } } = await worker.recognize(preprocessed);

        // Pass 2 — PSM 11: sparse/scattered text (catches logos, corner text, QR labels)
        await worker.setParameters({ tessedit_pageseg_mode: '11' as any });
        const { data: { text: text11 } } = await worker.recognize(preprocessed);
        await worker.setParameters({ tessedit_pageseg_mode: '6' as any }); // reset

        // Merge: union of unique non-empty lines from both passes
        const lines6 = new Set(text6.split('\n').map(l => l.trim()).filter(Boolean));
        const merged = [...lines6];
        text11.split('\n').map(l => l.trim()).filter(Boolean).forEach(l => {
          if (!lines6.has(l)) merged.push(l);
        });
        combinedText += merged.join('\n') + '\n';
      }

      const contact = parseContactFromText(combinedText);
      if (!hasUsefulInfo(contact)) throw new Error('NO_INFO');
      setPartialContact(contact);
      return contact;
    } finally {
      await worker.terminate();
    }
  };

  // ─── Orchestrator: try providers in waterfall order ─────────────────────────
  const scanCard = async (base64Images: string[]) => {
    if (isScanning) return;
    setIsScanning(true);
    setError(null);
    setPartialContact({});
    setActiveProvider(null);
    setStep('analyzing');

    // @ts-ignore
    const geminiKey = (import.meta as any).env?.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY;
    // @ts-ignore
    const groqKey = (import.meta as any).env?.VITE_GROQ_API_KEY;
    // @ts-ignore
    const openrouterKey = (import.meta as any).env?.VITE_OPENROUTER_API_KEY;

    const providers: Array<{ name: 'gemini' | 'groq' | 'openrouter' | 'tesseract'; fn: () => Promise<ContactInfo>; available: boolean }> = [
      { name: 'gemini',      fn: () => scanWithGemini(base64Images),      available: !!(geminiKey && geminiKey !== 'MY_GEMINI_API_KEY' && !geminiKey.includes('YOUR_API_KEY')) },
      { name: 'groq',        fn: () => scanWithGroq(base64Images),        available: !!(groqKey && !groqKey.includes('YOUR')) },
      { name: 'openrouter',  fn: () => scanWithOpenRouter(base64Images),  available: !!(openrouterKey && !openrouterKey.includes('YOUR')) },
      { name: 'tesseract',   fn: () => scanWithTesseract(base64Images),   available: true },
    ];

    let lastError = '';
    for (const provider of providers) {
      if (!provider.available) continue;
      setActiveProvider(provider.name);
      try {
        const result = await provider.fn();
        setContact(result);
        setStep('result');
        confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 }, colors: ['#10b981', '#3b82f6', '#f59e0b'] });
        setIsScanning(false);
        setPartialContact(null);
        setActiveProvider(null);
        return;
      } catch (err: any) {
        const msg = err?.message || '';
        console.warn(`[${provider.name}] failed:`, msg);
        if (msg === 'NO_KEY') continue; // silently skip — no key configured
        if (msg === 'QUOTA') {
          console.warn(`[${provider.name}] quota exceeded, trying next provider`);
          continue;
        }
        if (msg === 'NO_INFO') {
          lastError = 'Could not extract contact information from this card. Please ensure it is clear and well-lit.';
          continue; // try next provider — might do better
        }
        lastError = msg;
        continue;
      }
    }

    // All providers failed
    setError(lastError || 'All scanning methods failed. Please check your API keys in the .env file or try a clearer image.');
    setStep('review');
    setIsScanning(false);
    setPartialContact(null);
    setActiveProvider(null);
  };
  const reset = () => {
    setImages([]);
    setContact(null);
    setPartialContact(null);
    setError(null);
    setIsScanning(false);
    setStep('front');
    startCamera();
  };

  return (
    <div className="min-h-screen flex flex-col bg-stone-50 text-stone-900 font-sans selection:bg-emerald-100">
      {/* Header */}
      <header className="p-4 flex justify-between items-center border-b border-stone-200 bg-white/80 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-600 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-200">
            <ScanLine className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-xl tracking-tight">CardScan AI</h1>
            <p className="text-[10px] text-stone-500 font-medium uppercase tracking-widest">Intelligent Contact Capture</p>
          </div>
        </div>
        {(images.length > 0 || contact) && (
          <button
            onClick={reset}
            className="p-2 hover:bg-stone-100 rounded-full transition-colors text-stone-400 hover:text-stone-600"
          >
            <X className="w-6 h-6" />
          </button>
        )}
      </header>

      <main className="max-w-xl mx-auto p-3 pb-10">
        <AnimatePresence mode="wait">
          {(step === 'front' || step === 'back' || step === 'review') && !contact ? (
            <motion.div
              key="camera"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="space-y-3"
            >
              {(step === 'front' || step === 'back') && (
                <div className="relative aspect-[4/3] sm:aspect-[3/2] max-h-[40vh] sm:max-h-none bg-stone-900 rounded-2xl sm:rounded-3xl overflow-hidden shadow-2xl border-4 border-white">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />

                  {/* Flash Effect */}
                  <AnimatePresence>
                    {showFlash && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 bg-white z-20"
                      />
                    )}
                  </AnimatePresence>

                  {/* Guide Overlay */}
                  <div className="absolute inset-0 border-[40px] border-black/40 pointer-events-none">
                    <div className="w-full h-full border-2 border-white/50 rounded-lg flex items-center justify-center">
                      <div className="w-12 h-12 border-t-2 border-l-2 border-white absolute top-4 left-4" />
                      <div className="w-12 h-12 border-t-2 border-r-2 border-white absolute top-4 right-4" />
                      <div className="w-12 h-12 border-b-2 border-l-2 border-white absolute bottom-4 left-4" />
                      <div className="w-12 h-12 border-b-2 border-r-2 border-white absolute bottom-4 right-4" />
                    </div>
                  </div>

                  {!isCameraReady && !error && (
                    <div className="absolute inset-0 flex items-center justify-center bg-stone-900">
                      <Loader2 className="w-8 h-8 text-white animate-spin" />
                    </div>
                  )}

                  {error && images.length === 0 && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-stone-900/90 p-8 text-center">
                      <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
                      <p className="text-white text-sm mb-6">{error}</p>
                      <button
                        onClick={startCamera}
                        className="bg-white text-stone-900 px-6 py-2 rounded-xl font-bold text-sm hover:bg-stone-100 transition-colors flex items-center gap-2"
                      >
                        <Camera className="w-4 h-4" />
                        Retry Camera
                      </button>
                    </div>
                  )}
                </div>
              )}

              <div className="text-center space-y-1">
                {step === 'review' ? (
                  <h2 className="text-base font-semibold text-emerald-600 flex items-center justify-center gap-2">
                    <Check className="w-5 h-5" />
                    Captures Ready
                  </h2>
                ) : (
                  <h2 className="text-base font-semibold">
                    {step === 'front' ? 'Capture Front Side' : 'Capture Back Side'}
                  </h2>
                )}
                <p className="text-stone-500 text-xs">
                  {step === 'front'
                    ? 'Position the front of the card within the frame.'
                    : step === 'back'
                      ? 'Capture the back side if it has info.'
                      : 'Review your captures below before processing.'}
                </p>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-2 bg-red-50 border border-red-100 rounded-lg flex items-center gap-2 text-red-600 text-xs mx-4"
                  >
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    <p className="font-medium">{error}</p>
                  </motion.div>
                )}
              </div>

              <div className="flex flex-col items-center gap-3">
                {step === 'review' ? (
                  <button
                    onClick={() => scanCard(images)}
                    className="w-full bg-emerald-600 text-white py-4 rounded-2xl font-bold shadow-lg shadow-emerald-100 hover:bg-emerald-700 active:scale-[0.98] transition-all flex items-center justify-center gap-2"
                  >
                    <ScanLine className="w-5 h-5" />
                    Process Visiting Card
                  </button>
                ) : (
                  <>
                    {cooldown > 0 ? (
                      <div className="flex flex-col items-center gap-3">
                        <div className="w-14 h-14 bg-stone-100 rounded-full flex items-center justify-center border-4 border-stone-200">
                          <span className="text-stone-400 font-bold">{cooldown}s</span>
                        </div>
                        <p className="text-[10px] text-stone-400 font-bold uppercase tracking-widest">Cooling down...</p>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-3 w-full">
                        {/* Camera Capture Button */}
                        <button
                          onClick={captureImage}
                          disabled={!isCameraReady}
                          className="group relative w-14 h-14 bg-white rounded-full flex items-center justify-center shadow-xl border-4 border-stone-100 active:scale-95 transition-transform disabled:opacity-50"
                          title="Capture from camera"
                        >
                          <div className="w-10 h-10 bg-emerald-600 rounded-full flex items-center justify-center group-hover:bg-emerald-500 transition-colors">
                            <Camera className="text-white w-5 h-5" />
                          </div>
                        </button>

                        {/* Divider */}
                        <div className="flex items-center gap-3 w-full max-w-xs">
                          <div className="flex-1 h-px bg-stone-200" />
                          <span className="text-[10px] font-bold uppercase tracking-widest text-stone-400">or</span>
                          <div className="flex-1 h-px bg-stone-200" />
                        </div>

                        {/* Upload from Gallery Button */}
                        <button
                          onClick={() =>
                            step === 'front'
                              ? fileInputRef.current?.click()
                              : fileInputBackRef.current?.click()
                          }
                          className="flex items-center gap-2 px-5 py-2.5 bg-stone-100 hover:bg-stone-200 text-stone-700 font-bold text-sm rounded-xl transition-all active:scale-[0.97] border border-stone-200"
                          title="Upload from gallery"
                        >
                          <ImagePlus className="w-4 h-4 text-emerald-600" />
                          Upload from Gallery
                        </button>

                        {/* Hidden file inputs */}
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => handleGalleryUpload(e, 'front')}
                          aria-label="Upload front side of card from gallery"
                        />
                        <input
                          ref={fileInputBackRef}
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => handleGalleryUpload(e, 'back')}
                          aria-label="Upload back side of card from gallery"
                        />
                      </div>
                    )}

                    {step === 'back' && (
                      <button
                        onClick={skipBackSide}
                        className="text-stone-400 hover:text-stone-600 font-bold text-sm uppercase tracking-widest flex items-center gap-2"
                      >
                        Skip Back Side
                        <X className="w-4 h-4" />
                      </button>
                    )}
                  </>
                )}
              </div>

              {/* Scanning Tips - More compact */}
              <div className="bg-emerald-50/50 border border-emerald-100/50 rounded-2xl p-3 space-y-2">
                <p className="text-[10px] font-bold uppercase tracking-widest text-emerald-700 flex items-center gap-2">
                  <ScanLine className="w-3 h-3" />
                  Quick Tips
                </p>
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  <div className="flex items-center gap-1.5">
                    <div className="w-1 h-1 rounded-full bg-emerald-400" />
                    <p className="text-[10px] text-stone-600">Good lighting</p>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1 h-1 rounded-full bg-emerald-400" />
                    <p className="text-[10px] text-stone-600">No reflections</p>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1 h-1 rounded-full bg-emerald-400" />
                    <p className="text-[10px] text-stone-600">Steady hands</p>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1 h-1 rounded-full bg-emerald-400" />
                    <p className="text-[10px] text-stone-600">Plain background</p>
                  </div>
                </div>
              </div>

              {/* Captured Images Preview - NEW: Shown below tips */}
              {images.length > 0 && (
                <div className="space-y-3 pt-2">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-stone-400 px-1">Captured Sides</p>
                  <div className="grid grid-cols-2 gap-3">
                    {images.map((img, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="relative aspect-[3/2] bg-stone-200 rounded-xl overflow-hidden shadow-sm border-2 border-white"
                      >
                        <img src={img} alt={`Captured side ${idx + 1}`} className="w-full h-full object-cover" />
                        <div className="absolute top-2 left-2 bg-stone-900/60 text-white text-[8px] px-1.5 py-0.5 rounded-md backdrop-blur-sm font-bold uppercase">
                          {idx === 0 ? 'Front' : 'Back'}
                        </div>
                      </motion.div>
                    ))}
                    {images.length === 1 && step === 'back' && (
                      <div className="aspect-[3/2] bg-stone-100 rounded-xl border-2 border-dashed border-stone-200 flex flex-col items-center justify-center text-stone-400 p-4 text-center">
                        <UserPlus className="w-5 h-5 mb-1 opacity-50" />
                        <p className="text-[9px] font-medium leading-tight">Awaiting Back Side<br />(optional)</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          ) : step === 'analyzing' ? (
            <motion.div
              key="analyzing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              <div className="bg-white p-8 rounded-[2rem] shadow-xl shadow-stone-200/50 border border-stone-100 space-y-8 relative overflow-hidden">
                {/* Shimmer Effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-stone-50/50 to-transparent -translate-x-full animate-[shimmer_2s_infinite]" />

                <div className="space-y-3">
                  <p className="text-[10px] uppercase tracking-widest font-bold text-emerald-600 flex items-center gap-2">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Extracting Data...
                  </p>
                  {/* Active provider badge */}
                  {activeProvider && (
                    <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest border"
                      style={{
                        background: activeProvider === 'gemini' ? '#ecfdf5' : activeProvider === 'groq' ? '#eff6ff' : activeProvider === 'openrouter' ? '#faf5ff' : '#fefce8',
                        color: activeProvider === 'gemini' ? '#059669' : activeProvider === 'groq' ? '#2563eb' : activeProvider === 'openrouter' ? '#7c3aed' : '#ca8a04',
                        borderColor: activeProvider === 'gemini' ? '#a7f3d0' : activeProvider === 'groq' ? '#bfdbfe' : activeProvider === 'openrouter' ? '#ddd6fe' : '#fde68a',
                      }}
                    >
                      {activeProvider === 'tesseract' ? <Cpu className="w-3 h-3" /> : <Zap className="w-3 h-3" />}
                      {activeProvider === 'gemini' && 'Gemini AI'}
                      {activeProvider === 'groq' && 'Groq · LLaMA 4'}
                      {activeProvider === 'openrouter' && 'OpenRouter · LLaMA'}
                      {activeProvider === 'tesseract' && 'Offline OCR'}
                    </div>
                  )}

                  <div className="h-8 w-3/4 bg-stone-100 rounded-lg overflow-hidden flex items-center">
                    {partialContact?.firstName ? (
                      <span className="px-1 text-2xl font-bold tracking-tight text-stone-900">
                        {partialContact.firstName} {partialContact.lastName || ''}
                      </span>
                    ) : (
                      <div className="w-full h-full bg-stone-100 animate-pulse flex items-center px-3">
                        <span className="text-stone-300 text-sm font-medium">Waiting for AI...</span>
                      </div>
                    )}
                  </div>

                  <div className="h-4 w-1/2 bg-stone-50 rounded animate-pulse">
                    {partialContact?.title && (
                      <span className="px-1 text-stone-700 font-medium">{partialContact.title}</span>
                    )}
                  </div>

                  <div className="h-4 w-1/3 bg-stone-50 rounded animate-pulse">
                    {partialContact?.company && (
                      <span className="px-1 text-stone-500 font-medium">{partialContact.company}</span>
                    )}
                  </div>
                </div>

                <div className="grid gap-6">
                  {[
                    { icon: Phone, label: 'Landline', value: partialContact?.landlines?.[0] },
                    { icon: Smartphone, label: 'Mobile', value: partialContact?.mobiles?.[0] },
                    { icon: Mail, label: 'Email', value: partialContact?.email },
                    { icon: Globe, label: 'Website', value: partialContact?.website },
                    { icon: MapPin, label: 'Address', value: partialContact?.address },
                  ].map((field, idx) => (
                    <div key={idx} className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-stone-50 flex items-center justify-center text-stone-300">
                        <field.icon className="w-5 h-5" />
                      </div>
                      <div className="flex-1 space-y-1">
                        <p className="text-[10px] uppercase tracking-widest font-bold text-stone-300">{field.label}</p>
                        {field.value ? (
                          <motion.p
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="font-medium text-stone-900"
                          >
                            {field.value}
                          </motion.p>
                        ) : (
                          <div className="h-4 w-full bg-stone-50 rounded animate-pulse" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="text-center space-y-2">
                <p className="text-stone-500 text-sm font-medium">Reading {images.length} side{images.length > 1 ? 's' : ''} of your card...</p>
                <div className="w-full max-w-xs mx-auto bg-stone-100 h-1 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: activeProvider === 'tesseract' ? 12 : 5, ease: "linear" }}
                    className="h-full bg-emerald-500"
                  />
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-6"
            >
              {/* Captured Images Preview */}
              <div className="grid grid-cols-2 gap-4">
                {images.map((img, idx) => (
                  <div key={idx} className="relative aspect-[3/2] bg-stone-200 rounded-2xl overflow-hidden shadow-md border-2 border-white">
                    <img src={img} alt={`Card side ${idx + 1}`} className="w-full h-full object-cover" />
                    <div className="absolute top-2 left-2 bg-black/50 text-white text-[10px] px-2 py-0.5 rounded-full backdrop-blur-sm">
                      Side {idx + 1}
                    </div>
                  </div>
                ))}
              </div>

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-red-50 border border-red-100 p-4 rounded-2xl flex items-start gap-3 text-red-700"
                >
                  <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                  <div className="space-y-1">
                    <p className="font-semibold text-sm">Scanning Failed</p>
                    <p className="text-sm opacity-90">{error}</p>
                    <button
                      onClick={reset}
                      className="text-xs font-bold uppercase tracking-wider mt-2 hover:underline"
                    >
                      Try Again
                    </button>
                  </div>
                </motion.div>
              )}

              {contact && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white p-8 rounded-[2rem] shadow-xl shadow-stone-200/50 border border-stone-100 space-y-8"
                >
                  <div className="space-y-1">
                    <p className="text-[10px] uppercase tracking-widest font-bold text-emerald-600">Contact Name</p>
                    <h3 className="text-2xl font-bold tracking-tight text-stone-900">
                      {contact.firstName} {contact.lastName}
                    </h3>
                    {contact.title && <p className="text-stone-700 font-medium">{contact.title}</p>}
                    {contact.company && <p className="text-stone-500 font-medium">{contact.company}</p>}
                  </div>

                  <div className="grid gap-6">
                    {contact.landlines?.map((num, idx) => (
                      <div key={`landline-${idx}`} className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-stone-50 flex items-center justify-center text-stone-400">
                          <Phone className="w-5 h-5" />
                        </div>
                        <div>
                          <p className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Landline Number {contact.landlines.length > 1 ? idx + 1 : ''}</p>
                          <p className="font-medium">{num}</p>
                        </div>
                      </div>
                    ))}
                    {contact.mobiles?.map((num, idx) => (
                      <div key={`mobile-${idx}`} className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-stone-50 flex items-center justify-center text-stone-400">
                          <Smartphone className="w-5 h-5" />
                        </div>
                        <div>
                          <p className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Mobile Number {contact.mobiles.length > 1 ? idx + 1 : ''}</p>
                          <p className="font-medium">{num}</p>
                        </div>
                      </div>
                    ))}
                    {contact.email && (
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-stone-50 flex items-center justify-center text-stone-400">
                          <Mail className="w-5 h-5" />
                        </div>
                        <div>
                          <p className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Email</p>
                          <p className="font-medium">{contact.email}</p>
                        </div>
                      </div>
                    )}
                    {contact.website && (
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-stone-50 flex items-center justify-center text-stone-400">
                          <Globe className="w-5 h-5" />
                        </div>
                        <div>
                          <p className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Website</p>
                          <p className="font-medium">{contact.website}</p>
                        </div>
                      </div>
                    )}
                    {contact.address && (
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-stone-50 flex items-center justify-center text-stone-400">
                          <MapPin className="w-5 h-5" />
                        </div>
                        <div>
                          <p className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Address</p>
                          <p className="font-medium text-sm leading-relaxed">{contact.address}</p>
                        </div>
                      </div>
                    )}
                    {contact.notes && (
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-stone-50 flex items-center justify-center text-stone-400">
                          <FileText className="w-5 h-5" />
                        </div>
                        <div>
                          <p className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Notes</p>
                          <p className="font-medium text-sm leading-relaxed">{contact.notes}</p>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="pt-4 flex gap-3">
                    <button
                      onClick={() => downloadVCard(contact)}
                      className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-4 rounded-2xl flex items-center justify-center gap-2 shadow-lg shadow-emerald-200 transition-all active:scale-[0.98]"
                    >
                      <UserPlus className="w-5 h-5" />
                      Save to Contacts
                    </button>
                    <button
                      onClick={reset}
                      className="w-14 bg-stone-100 hover:bg-stone-200 text-stone-600 rounded-2xl flex items-center justify-center transition-all active:scale-[0.98]"
                    >
                      <RefreshCw className="w-5 h-5" />
                    </button>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Hidden Canvas for Capturing */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Footer Info */}
      <footer className="p-4 text-center mt-auto">
        <p className="text-[10px] text-stone-400 font-medium uppercase tracking-widest">
          Powered by Gemini AI • Secure & Private
        </p>
      </footer>
    </div>
  );
}
