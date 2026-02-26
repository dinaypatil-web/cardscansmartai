/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, RefreshCw, UserPlus, Download, Check, AlertCircle, Loader2, ScanLine, X, Globe, FileText, Mail, MapPin, Briefcase, Phone, Smartphone, Edit3, Plus, Minus, ArrowRightLeft } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import Tesseract from 'tesseract.js';
import { GoogleGenAI } from '@google/genai';
import { ContactInfo, parseContactFromText, parsePartialContact } from './contactParser';
import confetti from 'canvas-confetti';

// ContactInfo type is now imported from contactParser.ts

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
 * Resize and compress image for sending to Gemini API.
 * High quality for best OCR accuracy — Gemini handles large images well.
 */
const compressImageForAPI = (base64Str: string, maxWidth = 1600, maxHeight = 1600): Promise<string> => {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = base64Str;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      let width = img.width;
      let height = img.height;

      if (width > height) {
        if (width > maxWidth) {
          height *= maxWidth / width;
          width = maxWidth;
        }
      } else {
        if (height > maxHeight) {
          width *= maxHeight / height;
          height = maxHeight;
        }
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0, width, height);
      }
      resolve(canvas.toDataURL('image/jpeg', 0.92));
    };
  });
};

/**
 * Preprocess image for Tesseract.js fallback OCR.
 * Grayscale + contrast enhancement for better text extraction.
 */
const preprocessImageForOCR = (base64Str: string, maxWidth = 1600, maxHeight = 1600): Promise<string> => {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = base64Str;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      let width = img.width;
      let height = img.height;

      if (width > height) {
        if (width > maxWidth) { height *= maxWidth / width; width = maxWidth; }
      } else {
        if (height > maxHeight) { width *= maxHeight / height; height = maxHeight; }
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0, width, height);

        const imageData = ctx.getImageData(0, 0, width, height);
        const data = imageData.data;

        // Grayscale + contrast stretch
        let min = 255, max = 0;
        for (let i = 0; i < data.length; i += 4) {
          const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          data[i] = data[i + 1] = data[i + 2] = gray;
          if (gray < min) min = gray;
          if (gray > max) max = gray;
        }
        const range = max - min || 1;
        for (let i = 0; i < data.length; i += 4) {
          const val = Math.min(255, Math.max(0, ((data[i] - min) / range) * 255));
          data[i] = data[i + 1] = data[i + 2] = val;
        }
        ctx.putImageData(imageData, 0, 0);
      }
      resolve(canvas.toDataURL('image/png'));
    };
  });
};


// --- Components ---

export default function App() {
  const [images, setImages] = useState<string[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [contact, setContact] = useState<ContactInfo | null>(null);
  const [editContact, setEditContact] = useState<ContactInfo | null>(null);
  const [partialContact, setPartialContact] = useState<Partial<ContactInfo> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cooldown, setCooldown] = useState<number>(0);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [step, setStep] = useState<'front' | 'back' | 'analyzing' | 'result'>('front');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Helper to update a single field in editContact
  const updateField = (field: keyof ContactInfo, value: any) => {
    setEditContact(prev => prev ? { ...prev, [field]: value } : prev);
  };

  // Helper to update a phone number in a list
  const updatePhone = (type: 'mobiles' | 'landlines', index: number, value: string) => {
    setEditContact(prev => {
      if (!prev) return prev;
      const list = [...prev[type]];
      list[index] = value;
      return { ...prev, [type]: list };
    });
  };

  // Helper to add a new phone number
  const addPhone = (type: 'mobiles' | 'landlines') => {
    setEditContact(prev => {
      if (!prev) return prev;
      return { ...prev, [type]: [...prev[type], ''] };
    });
  };

  // Helper to remove a phone number
  const removePhone = (type: 'mobiles' | 'landlines', index: number) => {
    setEditContact(prev => {
      if (!prev) return prev;
      const list = prev[type].filter((_, i) => i !== index);
      return { ...prev, [type]: list };
    });
  };

  // Helper to toggle a phone between mobile and landline
  const togglePhoneType = (fromType: 'mobiles' | 'landlines', index: number) => {
    setEditContact(prev => {
      if (!prev) return prev;
      const toType = fromType === 'mobiles' ? 'landlines' : 'mobiles';
      const number = prev[fromType][index];
      const fromList = prev[fromType].filter((_, i) => i !== index);
      const toList = [...prev[toType], number];
      return { ...prev, [fromType]: fromList, [toType]: toList };
    });
  };

  // Initialize Camera
  useEffect(() => {
    if ((step === 'front' || step === 'back') && !contact) {
      startCamera();
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
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg');

        const newImages = [...images, dataUrl];
        setImages(newImages);

        if (step === 'front') {
          setStep('back');
        } else {
          setStep('analyzing');
          stopCamera();
          scanCard(newImages);
        }
      }
    }
  }, [images, step]);

  const skipBackSide = () => {
    setStep('analyzing');
    stopCamera();
    scanCard(images);
  };

  /**
   * Smart scanning pipeline:
   * PRIMARY: Send images directly to Gemini 2.0 Flash for OCR + extraction (best accuracy)
   * FALLBACK: Tesseract.js local OCR + regex parser (when API unavailable)
   */
  const scanCard = async (base64Images: string[]) => {
    setIsScanning(true);
    setError(null);
    setPartialContact({});
    setStep('analyzing');

    try {
      let finalContact: ContactInfo;
      const apiKey = process.env.GEMINI_API_KEY;

      if (apiKey && apiKey !== 'MY_GEMINI_API_KEY') {
        // PRIMARY: Gemini 2.0 Flash with direct image analysis
        try {
          finalContact = await scanWithGemini(base64Images, apiKey);
        } catch (geminiErr: any) {
          console.warn('Gemini failed, trying Tesseract fallback:', geminiErr.message);
          if (geminiErr.message?.includes('429') || geminiErr.message?.includes('RESOURCE_EXHAUSTED')) {
            throw geminiErr;
          }
          finalContact = await scanWithTesseract(base64Images);
        }
      } else {
        finalContact = await scanWithTesseract(base64Images);
      }

      const hasInfo = finalContact.firstName ||
        finalContact.lastName ||
        finalContact.company ||
        finalContact.email ||
        (finalContact.mobiles && finalContact.mobiles.length > 0) ||
        (finalContact.landlines && finalContact.landlines.length > 0);

      if (hasInfo) {
        setContact(finalContact);
        setEditContact({ ...finalContact });
        setStep('result');
        confetti({
          particleCount: 100,
          spread: 70,
          origin: { y: 0.6 },
          colors: ['#10b981', '#3b82f6', '#f59e0b']
        });
      } else {
        throw new Error("Could not extract any contact information from the card. Please ensure the card is clearly visible and well-lit.");
      }
    } catch (err: any) {
      console.error("Scanning error:", err);
      const errorMessage = err.message || JSON.stringify(err);

      if (errorMessage.includes('429') || errorMessage.includes('RESOURCE_EXHAUSTED')) {
        setError("Rate limit reached. Please wait a moment and try again.");
        setCooldown(30);
      } else if (errorMessage.includes("extract any contact information")) {
        setError(errorMessage);
      } else {
        setError("Failed to scan the card. Please ensure the image is clear, well-lit, and try again.");
      }
      setStep('front');
      setImages([]);
    } finally {
      setIsScanning(false);
      setPartialContact(null);
    }
  };

  /**
   * PRIMARY: Send images directly to Gemini 2.0 Flash.
   * Gemini handles both OCR and field extraction in one step — best accuracy.
   * Includes retry logic with exponential backoff and JSON error recovery.
   * Free tier: 1,500 requests/day, 15 RPM.
   */
  const scanWithGemini = async (base64Images: string[], apiKey: string): Promise<ContactInfo> => {
    const ai = new GoogleGenAI({ apiKey });
    const compressed = await Promise.all(base64Images.map(img => compressImageForAPI(img)));

    const imageParts = compressed.map(img => ({
      inlineData: {
        mimeType: "image/jpeg" as const,
        data: img.split(',')[1]
      }
    }));

    const prompt = `You are an expert OCR system specialized in reading business/visiting cards with perfect accuracy.

INSTRUCTIONS:
1. Carefully read EVERY piece of text on the card image(s), including small, faint, or stylized text.
2. The card may contain text in multiple languages (English, Hindi, Marathi, Gujarati, etc.). Extract ALL text regardless of language, but transliterate non-Latin names to English.
3. If there are multiple images, they are front and back of the SAME card — combine all information.

EXTRACTION RULES:
- "firstName" + "lastName": The PERSON's name (usually the largest/most prominent text). NEVER put the company name here.
- "company": The business/organization name. Look for suffixes like Pvt. Ltd., Inc., LLC, LLP, Corp., Solutions, Enterprises, Industries, Group, etc.
- "title": The person's job title or designation (e.g., Director, Manager, CEO, Proprietor, Partner).
- "mobiles": Phone numbers labeled Mobile/Cell/M/Mob/WhatsApp, OR Indian 10-digit numbers starting with 6/7/8/9. Include country code if shown (e.g., +91).
- "landlines": Numbers labeled Tel/Phone/Ph/Office/Fax/Land, OR numbers with STD codes (e.g., 022-XXXXXXXX). Include STD/area code.
- "email": Full email address.
- "website": Full URL (include http/https if shown, otherwise just the domain).
- "address": Complete postal address including building, street, area, city, state, and PIN/ZIP code. Combine address fragments from multiple lines.
- "notes": Any other useful info: GST number, PAN, social media handles (@twitter, LinkedIn URL), taglines, certifications, or additional details.

QUALITY RULES:
- Read numbers VERY carefully — do not confuse 0/O, 1/l/I, 5/S, 8/B, 6/G.
- Preserve exact phone number formatting with hyphens/spaces as shown on card.
- If a field is not found, use empty string "" or empty array [].
- Do NOT guess or hallucinate information that is not visible on the card.

Return ONLY a valid JSON object:
{
  "firstName": "string",
  "lastName": "string",
  "title": "string",
  "company": "string",
  "landlines": ["string"],
  "mobiles": ["string"],
  "email": "string",
  "website": "string",
  "address": "string",
  "notes": "string"
}`;

    // Retry logic with exponential backoff
    const MAX_RETRIES = 2;
    let lastError: any = null;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        if (attempt > 0) {
          const delay = Math.pow(2, attempt) * 1000; // 2s, 4s
          console.log(`Gemini retry attempt ${attempt}, waiting ${delay}ms...`);
          await new Promise(r => setTimeout(r, delay));
        }

        // Show partial results to user during analysis
        setPartialContact({ notes: attempt > 0 ? 'Retrying scan...' : undefined });

        const response = await ai.models.generateContent({
          model: 'gemini-2.0-flash',
          contents: {
            parts: [
              { text: prompt },
              ...imageParts
            ]
          },
          config: {
            responseMimeType: 'application/json',
          }
        });

        const text = response.text || '{}';
        let parsed: any;

        // Robust JSON parsing with error recovery
        try {
          parsed = JSON.parse(text);
        } catch (jsonErr) {
          console.warn('Gemini returned malformed JSON, attempting recovery...', text.substring(0, 200));
          // Try to extract JSON from the response text
          const jsonMatch = text.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            try {
              parsed = JSON.parse(jsonMatch[0]);
            } catch {
              throw new Error('Could not parse Gemini response as JSON');
            }
          } else {
            throw new Error('No JSON found in Gemini response');
          }
        }

        const contact: ContactInfo = {
          firstName: parsed.firstName || '',
          lastName: parsed.lastName || '',
          title: parsed.title || '',
          company: parsed.company || '',
          email: parsed.email || '',
          website: parsed.website || '',
          address: parsed.address || '',
          notes: parsed.notes || '',
          landlines: Array.isArray(parsed.landlines) ? parsed.landlines.filter(Boolean) : [],
          mobiles: Array.isArray(parsed.mobiles) ? parsed.mobiles.filter(Boolean) : [],
        };

        // Stream partial results to UI
        setPartialContact(contact);

        return contact;
      } catch (err: any) {
        lastError = err;
        const msg = err.message || '';
        // Only retry on rate limit errors
        if ((msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED')) && attempt < MAX_RETRIES) {
          continue;
        }
        throw err;
      }
    }

    throw lastError;
  };

  /**
   * FALLBACK: Tesseract.js local OCR + regex parser.
   * Used when Gemini API is unavailable or rate-limited.
   */
  const scanWithTesseract = async (base64Images: string[]): Promise<ContactInfo> => {
    const processedImages = await Promise.all(base64Images.map(img => preprocessImageForOCR(img)));
    let allText = '';

    for (const img of processedImages) {
      const result = await Tesseract.recognize(img, 'eng', {
        logger: (info: any) => {
          if (info.status === 'recognizing text' && info.progress > 0.3 && allText) {
            const partial = parsePartialContact(allText);
            setPartialContact(prev => ({ ...prev, ...partial }));
          }
        }
      });
      allText += result.data.text + '\n';
      const partial = parsePartialContact(allText);
      setPartialContact(prev => ({ ...prev, ...partial }));
    }

    console.log('Tesseract OCR text:', allText);
    return parseContactFromText(allText);
  };

  const reset = () => {
    setImages([]);
    setContact(null);
    setEditContact(null);
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
          {(step === 'front' || step === 'back') && !contact ? (
            <motion.div
              key="camera"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="space-y-3"
            >
              <div className="relative aspect-[4/3] sm:aspect-[3/2] max-h-[40vh] sm:max-h-none bg-stone-900 rounded-2xl sm:rounded-3xl overflow-hidden shadow-2xl border-4 border-white">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full h-full object-cover"
                />

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

              <div className="text-center space-y-1">
                <h2 className="text-base font-semibold">
                  {step === 'front' ? 'Capture Front Side' : 'Capture Back Side'}
                </h2>
                <p className="text-stone-500 text-xs">
                  {step === 'front'
                    ? 'Position the front of the card within the frame.'
                    : 'Capture the back side if it has info.'}
                </p>
              </div>

              <div className="flex flex-col items-center gap-3">
                {cooldown > 0 ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="w-14 h-14 bg-stone-100 rounded-full flex items-center justify-center border-4 border-stone-200">
                      <span className="text-stone-400 font-bold">{cooldown}s</span>
                    </div>
                    <p className="text-[10px] text-stone-400 font-bold uppercase tracking-widest">Cooling down...</p>
                  </div>
                ) : (
                  <button
                    onClick={captureImage}
                    disabled={!isCameraReady}
                    className="group relative w-14 h-14 bg-white rounded-full flex items-center justify-center shadow-xl border-4 border-stone-100 active:scale-95 transition-transform disabled:opacity-50"
                  >
                    <div className="w-10 h-10 bg-emerald-600 rounded-full flex items-center justify-center group-hover:bg-emerald-500 transition-colors">
                      <Camera className="text-white w-5 h-5" />
                    </div>
                  </button>
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

                  <div className="h-8 w-3/4 bg-stone-100 rounded-lg animate-pulse">
                    {partialContact?.firstName && (
                      <span className="px-1 text-2xl font-bold tracking-tight text-stone-900">
                        {partialContact.firstName} {partialContact.lastName || ''}
                      </span>
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
                    transition={{ duration: 8, ease: "linear" }}
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

              {editContact && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white p-6 sm:p-8 rounded-[2rem] shadow-xl shadow-stone-200/50 border border-stone-100 space-y-6"
                >
                  {/* Header */}
                  <div className="flex items-center gap-3 pb-2 border-b border-stone-100">
                    <div className="w-9 h-9 bg-amber-50 rounded-xl flex items-center justify-center">
                      <Edit3 className="w-4 h-4 text-amber-600" />
                    </div>
                    <div>
                      <p className="font-bold text-sm text-stone-900">Verify & Edit</p>
                      <p className="text-[10px] text-stone-400">Review the extracted data. Tap any field to correct it.</p>
                    </div>
                  </div>

                  {/* Name Fields */}
                  <div className="space-y-2">
                    <p className="text-[10px] uppercase tracking-widest font-bold text-emerald-600">Full Name</p>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-[10px] text-stone-400 font-medium">First Name</label>
                        <input
                          type="text"
                          value={editContact.firstName}
                          onChange={e => updateField('firstName', e.target.value)}
                          placeholder="First name"
                          className="w-full mt-0.5 px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] text-stone-400 font-medium">Last Name</label>
                        <input
                          type="text"
                          value={editContact.lastName}
                          onChange={e => updateField('lastName', e.target.value)}
                          placeholder="Last name"
                          className="w-full mt-0.5 px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Title & Company */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Briefcase className="w-3.5 h-3.5 text-stone-400" />
                        <label className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Designation</label>
                      </div>
                      <input
                        type="text"
                        value={editContact.title || ''}
                        onChange={e => updateField('title', e.target.value)}
                        placeholder="Job title / Designation"
                        className="w-full px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all"
                      />
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Briefcase className="w-3.5 h-3.5 text-stone-400" />
                        <label className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Company</label>
                      </div>
                      <input
                        type="text"
                        value={editContact.company || ''}
                        onChange={e => updateField('company', e.target.value)}
                        placeholder="Company / Organization"
                        className="w-full px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all"
                      />
                    </div>
                  </div>

                  {/* Phone Numbers Section */}
                  <div className="space-y-3">
                    {/* Landlines */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Phone className="w-3.5 h-3.5 text-stone-400" />
                          <label className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Landline Numbers</label>
                        </div>
                        <button
                          onClick={() => addPhone('landlines')}
                          className="p-1 rounded-lg hover:bg-emerald-50 text-emerald-500 hover:text-emerald-600 transition-colors"
                          title="Add landline"
                        >
                          <Plus className="w-4 h-4" />
                        </button>
                      </div>
                      {editContact.landlines.map((num, idx) => (
                        <div key={`landline-${idx}`} className="flex items-center gap-2">
                          <input
                            type="tel"
                            value={num}
                            onChange={e => updatePhone('landlines', idx, e.target.value)}
                            placeholder="Landline number"
                            className="flex-1 px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all"
                          />
                          <button
                            onClick={() => togglePhoneType('landlines', idx)}
                            className="p-2 rounded-xl bg-blue-50 hover:bg-blue-100 text-blue-500 transition-colors"
                            title="Move to Mobile"
                          >
                            <ArrowRightLeft className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => removePhone('landlines', idx)}
                            className="p-2 rounded-xl bg-red-50 hover:bg-red-100 text-red-400 transition-colors"
                            title="Remove"
                          >
                            <Minus className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      ))}
                      {editContact.landlines.length === 0 && (
                        <p className="text-xs text-stone-300 italic pl-1">No landline numbers detected</p>
                      )}
                    </div>

                    {/* Mobiles */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Smartphone className="w-3.5 h-3.5 text-stone-400" />
                          <label className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Mobile Numbers</label>
                        </div>
                        <button
                          onClick={() => addPhone('mobiles')}
                          className="p-1 rounded-lg hover:bg-emerald-50 text-emerald-500 hover:text-emerald-600 transition-colors"
                          title="Add mobile"
                        >
                          <Plus className="w-4 h-4" />
                        </button>
                      </div>
                      {editContact.mobiles.map((num, idx) => (
                        <div key={`mobile-${idx}`} className="flex items-center gap-2">
                          <input
                            type="tel"
                            value={num}
                            onChange={e => updatePhone('mobiles', idx, e.target.value)}
                            placeholder="Mobile number"
                            className="flex-1 px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all"
                          />
                          <button
                            onClick={() => togglePhoneType('mobiles', idx)}
                            className="p-2 rounded-xl bg-blue-50 hover:bg-blue-100 text-blue-500 transition-colors"
                            title="Move to Landline"
                          >
                            <ArrowRightLeft className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => removePhone('mobiles', idx)}
                            className="p-2 rounded-xl bg-red-50 hover:bg-red-100 text-red-400 transition-colors"
                            title="Remove"
                          >
                            <Minus className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      ))}
                      {editContact.mobiles.length === 0 && (
                        <p className="text-xs text-stone-300 italic pl-1">No mobile numbers detected</p>
                      )}
                    </div>
                  </div>

                  {/* Email */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Mail className="w-3.5 h-3.5 text-stone-400" />
                      <label className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Email</label>
                    </div>
                    <input
                      type="email"
                      value={editContact.email || ''}
                      onChange={e => updateField('email', e.target.value)}
                      placeholder="email@example.com"
                      className="w-full px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all"
                    />
                  </div>

                  {/* Website */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Globe className="w-3.5 h-3.5 text-stone-400" />
                      <label className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Website</label>
                    </div>
                    <input
                      type="url"
                      value={editContact.website || ''}
                      onChange={e => updateField('website', e.target.value)}
                      placeholder="www.example.com"
                      className="w-full px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all"
                    />
                  </div>

                  {/* Address */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <MapPin className="w-3.5 h-3.5 text-stone-400" />
                      <label className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Address</label>
                    </div>
                    <textarea
                      value={editContact.address || ''}
                      onChange={e => updateField('address', e.target.value)}
                      placeholder="Full postal address"
                      rows={2}
                      className="w-full px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all resize-none"
                    />
                  </div>

                  {/* Notes */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <FileText className="w-3.5 h-3.5 text-stone-400" />
                      <label className="text-[10px] uppercase tracking-widest font-bold text-stone-400">Notes</label>
                    </div>
                    <textarea
                      value={editContact.notes || ''}
                      onChange={e => updateField('notes', e.target.value)}
                      placeholder="GST, PAN, social media, or other notes"
                      rows={2}
                      className="w-full px-3 py-2.5 border border-stone-200 rounded-xl text-sm font-medium text-stone-900 bg-stone-50/50 focus:bg-white focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 outline-none transition-all resize-none"
                    />
                  </div>

                  {/* Action Buttons */}
                  <div className="pt-2 flex gap-3">
                    <button
                      onClick={() => downloadVCard(editContact)}
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
          Powered by Gemini 2.0 Flash AI • Smart & Accurate
        </p>
      </footer>
    </div>
  );
}
