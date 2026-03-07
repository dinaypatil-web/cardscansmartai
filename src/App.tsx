/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, RefreshCw, UserPlus, Download, Check, AlertCircle, Loader2, ScanLine, X, Globe, FileText, Mail, MapPin, Briefcase, Phone, Smartphone } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI, Type, ThinkingLevel } from "@google/genai";
import confetti from 'canvas-confetti';

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

const resizeImage = (base64Str: string, maxWidth = 800, maxHeight = 800): Promise<string> => {
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
      // Balanced quality for readability vs payload size
      resolve(canvas.toDataURL('image/jpeg', 0.7));
    };
  });
};

/**
 * Preprocess image for Tesseract.js fallback OCR.
 * Grayscale + adaptive thresholding for better text extraction.
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

        // Grayscale + Denoise + Contrast Stretch + Adaptive Thresholding (Simple)
        let min = 255, max = 0;
        for (let i = 0; i < data.length; i += 4) {
          // Weighted grayscale for better contrast
          const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          data[i] = data[i + 1] = data[i + 2] = gray;
          if (gray < min) min = gray;
          if (gray > max) max = gray;
        }

        const range = max - min || 1;
        const threshold = min + (range * 0.45); // Approximate adaptive threshold

        for (let i = 0; i < data.length; i += 4) {
          // Contrast stretch
          let val = ((data[i] - min) / range) * 255;

          // Simple Binarization (Threshholding) for Tesseract
          // We provide a slight gradient for "soft" binarization which Tesseract often likes better than pure salt/pepper
          val = val > threshold ? 255 : Math.max(0, val - 20);

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
  const [partialContact, setPartialContact] = useState<Partial<ContactInfo> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cooldown, setCooldown] = useState<number>(0);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [step, setStep] = useState<'front' | 'back' | 'review' | 'analyzing' | 'result'>('front');
  const [showFlash, setShowFlash] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

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

  const scanCard = async (base64Images: string[], retryCount = 0) => {
    if (isScanning && retryCount === 0) return; // Guard against multiple clicks
    setIsScanning(true);
    setError(null);
    setPartialContact({});
    setStep('analyzing');

    try {
      // @ts-ignore - Vite environment variable injection
      const apiKey = (import.meta as any).env?.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY;

      console.log("AI starting scan with images count:", base64Images.length);
      if (apiKey) {
        console.log("API Key found (starts with):", apiKey.substring(0, 4) + "...");
      } else {
        console.warn("API Key is missing or undefined.");
      }

      if (!apiKey || apiKey === "MY_GEMINI_API_KEY" || apiKey.includes("YOUR_API_KEY")) {
        throw new Error("Gemini API Key is missing. Please set VITE_GEMINI_API_KEY in your project environment variables.");
      }

      const ai = new GoogleGenAI({ apiKey });
      const compressedImages = await Promise.all(base64Images.map(img => resizeImage(img)));

      const prompt = `You are an expert OCR system specialized in reading business/visiting cards with 100% accuracy.

INSTRUCTIONS:
1. Carefully read EVERY piece of text on the card image(s), including small, faint, or stylized text.
2. The card may contain text in multiple languages (English, Hindi, Marathi, Gujarati, etc.). Extract ALL text regardless of language. If a name is in a non-Latin script, provide the Latin transliteration if possible, but prioritize accuracy.
3. If there are multiple images, they represent the front and back of the SAME card — combine all information intelligently.
4. Distinguish between brand/company names and personal names. If a person's name is also part of the company name (e.g., "John Doe Plumbing"), "John Doe" is the name and "John Doe Plumbing" is the company.

EXTRACTION RULES:
- "firstName" + "lastName": The PERSON's name. Look for honorifics (Mr., Dr., Adv.) as hints.
- "company": The legal or brand name of the business. Look for suffixes like Pvt. Ltd., LLC, Inc., etc.
- "title": Job title or designation. Be precise (e.g., "Founder & CEO" rather than just "CEO").
- "mobiles": Primarily for personal/mobile numbers. Look for "M:", "Mob:", or 10-digit Indian numbers starting with 6-9.
- "landlines": Office, desk, or fax numbers. Look for "T:", "Ph:", "Off:", or numbers with area/STD codes.
- "email": Comprehensive extraction. Look for "@" symbol.
- "website": Full URL or domain name.
- "address": Complete postal address. Combine all related fragments (Building, Street, Area, City, State, PIN).
- "notes": Extract GSTIN, PAN, Udyam numbers, social media handles (@handle), slogans/taglines, or any other professional certifications mentioned.

QUALITY RULES:
- NEVER confuse '0' (zero) with 'O' (letter) in phone numbers or IDs.
- Preserve exact formatting for IDs like GST/PAN.
- If a field is truly missing, return "" or [].
- DO NOT hallucinate. If text is illegible, leave the field empty.

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

      const imageParts = compressedImages.map(img => ({
        inlineData: {
          mimeType: "image/jpeg" as const,
          data: img.split(',')[1]
        }
      }));

      const stream = await ai.models.generateContentStream({
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [
              { text: prompt },
              ...imageParts
            ]
          }
        ],
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              firstName: { type: Type.STRING },
              lastName: { type: Type.STRING },
              title: { type: Type.STRING },
              company: { type: Type.STRING },
              landlines: { type: Type.ARRAY, items: { type: Type.STRING } },
              mobiles: { type: Type.ARRAY, items: { type: Type.STRING } },
              email: { type: Type.STRING },
              website: { type: Type.STRING },
              address: { type: Type.STRING },
              notes: { type: Type.STRING }
            }
          }
        }
      });

      let fullText = "";
      // @ts-ignore - response object structure
      for await (const chunk of stream.stream) {
        fullText += (chunk.text || "");

        // Try to parse partial JSON to show progress
        try {
          const cleaned = fullText.trim();
          if (cleaned.startsWith('{')) {
            const result: any = {};
            const fields = ['firstName', 'lastName', 'title', 'company', 'email', 'website', 'address', 'notes'];
            fields.forEach(field => {
              const regex = new RegExp(`"${field}"\\s*:\\s*"([^"]*)"`, 'g');
              const match = [...cleaned.matchAll(regex)].pop();
              if (match) result[field] = match[1];
            });

            ['landlines', 'mobiles'].forEach(field => {
              const regex = new RegExp(`"${field}"\\s*:\\s*\\[([^\\]]*)\\]`, 'g');
              const match = [...cleaned.matchAll(regex)].pop();
              if (match) {
                try {
                  const val = match[1].trim();
                  if (val === '') {
                    result[field] = [];
                  } else {
                    result[field] = val.split(',').map(s => s.trim().replace(/^"/, '').replace(/"$/, '').replace(/^'/, '').replace(/'$/, '')).filter(Boolean);
                  }
                } catch {
                  // Fallback
                }
              }
            });

            if (Object.keys(result).length > 0) {
              setPartialContact(prev => ({ ...prev, ...result }));
            }
          }
        } catch (e) {
          // Ignore partial parse errors
        }
      }

      const finalResult = JSON.parse(fullText || "{}");

      // Check if we got ANY useful information
      const hasInfo = finalResult.firstName ||
        finalResult.lastName ||
        finalResult.company ||
        finalResult.email ||
        (finalResult.mobiles && finalResult.mobiles.length > 0) ||
        (finalResult.landlines && finalResult.landlines.length > 0);

      if (hasInfo) {
        const sanitizedContact: ContactInfo = {
          firstName: finalResult.firstName || "",
          lastName: finalResult.lastName || "",
          title: finalResult.title || "",
          company: finalResult.company || "",
          email: finalResult.email || "",
          website: finalResult.website || "",
          address: finalResult.address || "",
          notes: finalResult.notes || "",
          landlines: Array.isArray(finalResult.landlines) ? finalResult.landlines.filter(Boolean) : [],
          mobiles: Array.isArray(finalResult.mobiles) ? finalResult.mobiles.filter(Boolean) : []
        };

        setContact(sanitizedContact);
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

      if (retryCount < 2 && (
        errorMessage.includes('xhr error') ||
        errorMessage.includes('500') ||
        errorMessage.includes('fetch') ||
        errorMessage.includes('Rpc failed') ||
        errorMessage.includes('429') ||
        errorMessage.includes('RESOURCE_EXHAUSTED')
      )) {
        const delay = Math.pow(2, retryCount + 1) * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
        return scanCard(base64Images, retryCount + 1);
      }

      if (errorMessage.includes('429') || errorMessage.includes('RESOURCE_EXHAUSTED')) {
        setError("API Quota Exceeded. The free tier limit has been reached. Please wait a moment before trying again.");
        setCooldown(60);
      } else if (errorMessage.includes("extract any contact information")) {
        setError(errorMessage);
      } else {
        setError(errorMessage.includes('Rpc failed')
          ? "The AI service is currently busy. Please try again."
          : "Failed to scan the card. Please ensure the image is clear and try again.");
      }
      setStep('review'); // Stay on review so user can see error and retry
    } finally {
      setIsScanning(false);
      setPartialContact(null);
    }
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
                    transition={{ duration: 4, ease: "linear" }}
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
