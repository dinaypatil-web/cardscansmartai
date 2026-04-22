/**
 * Contact Parser v2 - Enhanced extraction of structured contact information from raw OCR text.
 * Uses scoring-based classification, multi-pass analysis, and robust heuristics.
 */

export interface ContactInfo {
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

// ============================================================
// REGEX PATTERNS
// ============================================================

const EMAIL_REGEX = /[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g;

/**
 * Phone regex v3 — handles Indian STD, international, toll-free, and mobile formats.
 * Examples matched:
 *   +91 98765 43210  |  +91-9876543210  |  9876543210
 *   (022) 2345-6789  |  022-23456789    |  0261-234567
 *   1800-XXX-XXXX (toll-free)           |  +1 (800) 123-4567
 */
const PHONE_REGEX = /(?:(?:\+|00)\s*\d{1,3}[\s.\-]?)?(?:\(?0?\d{2,5}\)?[\s.\-]?)?\d{2,5}[\s.\-]?\d{2,6}(?:[\s.\-]?\d{2,4})?(?:\s*(?:\/|,|Ext\.?|x)\s*\d{1,6})*/g;

// Website regex — matches URLs and domain patterns
const WEBSITE_REGEX = /(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?(?:\.[a-zA-Z]{2,})+(?:\/[^\s,)]*)?/gi;

// Indian PIN code: 6 digits starting with 1-9 (not 0)
const PIN_CODE_REGEX = /\b[1-9]\d{5}\b/;

// Indian GST / PAN / Udyam identifiers — captured into notes
const GST_REGEX = /\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b/g;
const PAN_REGEX = /\b[A-Z]{5}\d{4}[A-Z]\b/g;
const UDYAM_REGEX = /UDYAM[\-\s]?[A-Z]{2}[\-\s]?\d{2}[\-\s]?\d{7}/gi;

// ============================================================
// KEYWORD DATABASES
// ============================================================

// Job title keywords with weights (higher = more confident)
const JOB_TITLE_PATTERNS: Array<{ pattern: RegExp; weight: number }> = [
  // C-suite and leadership
  { pattern: /\b(?:CEO|CTO|CFO|COO|CMO|CIO|CHRO)\b/i, weight: 10 },
  { pattern: /\bchief\s+\w+\s+officer\b/i, weight: 10 },
  { pattern: /\bmanaging\s+director\b/i, weight: 10 },
  { pattern: /\bgeneral\s+manager\b/i, weight: 9 },
  { pattern: /\bvice\s+president\b/i, weight: 9 },
  { pattern: /\b(?:president|chairman|chairperson)\b/i, weight: 9 },

  // Founders and owners
  { pattern: /\b(?:founder|co[\-\s]?founder|proprietor|owner|partner)\b/i, weight: 9 },

  // Director/Manager level
  { pattern: /\b(?:sr\.?|senior|junior|jr\.?|asst\.?|asstt\.?|assistant|associate|deputy)\s+\w+/i, weight: 5 },
  { pattern: /\bdirector\b/i, weight: 8 },
  { pattern: /\bmanager\b/i, weight: 7 },
  { pattern: /\bhead\s+(?:of\s+)?\w+/i, weight: 7 },
  { pattern: /\blead\b/i, weight: 6 },
  { pattern: /\bsupervisor\b/i, weight: 6 },

  // Professional roles — specific compound titles
  { pattern: /\b(?:software|hardware|mechanical|electrical|civil|chemical)\s+engineer\b/i, weight: 8 },
  { pattern: /\b(?:project|product|program|operations|sales|marketing|business|brand)\s+manager\b/i, weight: 8 },

  // Professional roles — single keywords
  { pattern: /\bengineer(?:ing)?\b/i, weight: 6 },
  { pattern: /\bdeveloper\b/i, weight: 6 },
  { pattern: /\barchitect\b/i, weight: 6 },
  { pattern: /\bdesigner\b/i, weight: 6 },
  { pattern: /\banalyst\b/i, weight: 6 },
  { pattern: /\bconsultant\b/i, weight: 6 },
  { pattern: /\bcoordinator\b/i, weight: 5 },
  { pattern: /\bspecialist\b/i, weight: 5 },
  { pattern: /\bexecutive\b/i, weight: 5 },
  { pattern: /\badvisor\b/i, weight: 5 },
  { pattern: /\badministrator\b/i, weight: 5 },
  { pattern: /\baccountant\b/i, weight: 6 },
  { pattern: /\badvocate\b/i, weight: 7 },
  { pattern: /\battorney\b/i, weight: 7 },
  { pattern: /\blawyer\b/i, weight: 7 },
  { pattern: /\bsolicitor\b/i, weight: 7 },
  { pattern: /\b(?:doctor|physician|surgeon)\b/i, weight: 7 },
  { pattern: /\b(?:professor|prof\.)\b/i, weight: 7 },
  { pattern: /\bteacher\b/i, weight: 5 },
  { pattern: /\bprincipal\b/i, weight: 5 },
  { pattern: /\bsecretary\b/i, weight: 5 },
  { pattern: /\btreasurer\b/i, weight: 5 },
  { pattern: /\breceptionist\b/i, weight: 5 },
  { pattern: /\btechnician\b/i, weight: 5 },
  { pattern: /\binspector\b/i, weight: 5 },
  { pattern: /\bofficer\b/i, weight: 4 },
  { pattern: /\bassistant\b/i, weight: 3 },

  // Dept keywords alone (lower weight — could be part of address)
  { pattern: /\b(?:sales|marketing|operations|finance|accounts|admin|hr|procurement)\b/i, weight: 2 },
];

// Company indicators with weights
const COMPANY_PATTERNS: Array<{ pattern: RegExp; weight: number }> = [
  // Strong legal suffixes
  { pattern: /\b(?:pvt|private)\s*\.?\s*(?:ltd|limited)\b/i, weight: 10 },
  { pattern: /\bopc\s*pvt\b/i, weight: 10 },
  { pattern: /\b(?:llp|lLP)\b/, weight: 9 },
  { pattern: /\bltd\.?\b/i, weight: 9 },
  { pattern: /\blimited\b/i, weight: 9 },
  { pattern: /\bllc\b/i, weight: 9 },
  { pattern: /\binc\.?\b/i, weight: 9 },
  { pattern: /\bincorporated\b/i, weight: 9 },
  { pattern: /\bcorp\.?\b/i, weight: 9 },
  { pattern: /\bcorporation\b/i, weight: 9 },
  { pattern: /\b(?:co|company)\.?\b/i, weight: 6 },
  { pattern: /\bproprietorship\b/i, weight: 9 },
  { pattern: /\bpartnership\b/i, weight: 8 },

  // Business entity types
  { pattern: /\bgroup\s+of\s+(?:companies|industries)\b/i, weight: 10 },
  { pattern: /\bgroup\b/i, weight: 5 },
  { pattern: /\b(?:enterprises?|ventures?|holdings?)\b/i, weight: 7 },
  { pattern: /\binfra(?:structure)?\b/i, weight: 6 },

  // Service/Industry keywords
  { pattern: /\b(?:solutions?|services?|technologies|tech)\b/i, weight: 5 },
  { pattern: /\b(?:industries|associates|consultants|consulting)\b/i, weight: 6 },
  { pattern: /\b(?:international|global|worldwide|overseas)\b/i, weight: 3 },
  { pattern: /\b(?:foundation|institute|academy|school|college|university)\b/i, weight: 5 },
  { pattern: /\b(?:labs?|studio|systems?)\b/i, weight: 4 },
  { pattern: /\b(?:infotech|infosystems|softech|softtech|infocomm)\b/i, weight: 7 },
  { pattern: /\b(?:traders?|trading|exports?|imports?|distributors?|suppliers?)\b/i, weight: 6 },
  { pattern: /\b(?:hospital|clinic|pharmacy|medical|diagnostics|healthcare)\b/i, weight: 5 },
  { pattern: /\b(?:builders?|construction|realty|real\s*estate|developers?|infra)\b/i, weight: 5 },
  { pattern: /\b(?:motors?|automobiles?|auto|vehicles?)\b/i, weight: 4 },
  { pattern: /\b(?:jewellers?|jewelers?|textiles?|garments?|fashion|boutique)\b/i, weight: 5 },
  { pattern: /\b(?:publishers?|printing|press|media|publications?)\b/i, weight: 5 },
  { pattern: /\b(?:agency|agencies|marketing|advertising|promotions?)\b/i, weight: 5 },
  { pattern: /\b(?:logistics?|transport|cargo|shipping|courier|freight)\b/i, weight: 6 },
  { pattern: /\b(?:foods?|catering|restaurant|hotel|hospitality|bakery)\b/i, weight: 5 },
  { pattern: /\bmidc\b/i, weight: 4 },
];

// Address keywords
const ADDRESS_KEYWORDS = [
  'road', 'rd\\.?', 'street', 'st\\.?', 'avenue', 'ave',
  'lane', 'gali', 'marg', 'path', 'chowk', 'square',
  'building', 'bldg', 'tower', 'complex', 'arcade', 'plaza', 'mall', 'centre', 'center',
  'floor', 'flr', 'suite', 'ste', 'flat', 'plot', 'shop', 'office', 'off', 'cabin', 'room',
  'sector', 'block', 'phase', 'wing', 'industrial', 'estate', 'area',
  'nagar', 'colony', 'society', 'housing', 'residency', 'enclave', 'vihar', 'puram',
  'estate', 'layout', 'extension', 'extn',
  'village', 'taluk', 'taluka', 'tehsil', 'mandal',
  'city', 'town', 'district', 'dist',
  'state', 'province', 'country',
  'near', 'opp\\.?', 'opposite', 'beside', 'behind', 'next\\s+to', 'adjacent',
  'post\\s+office', 'p\\.?o\\.?',
  'pin', 'zip', 'landmark',
];

// Indian and major international cities/states for address detection
const CITY_NAMES = [
  // Top metros
  'mumbai', 'delhi', 'new delhi', 'bangalore', 'bengaluru', 'chennai', 'kolkata', 'hyderabad',
  // Maharashtra
  'pune', 'nagpur', 'nashik', 'aurangabad', 'solapur', 'amravati', 'navi mumbai',
  'thane', 'kalyan', 'dombivli', 'vasai', 'virar', 'panvel', 'raigad',
  // Gujarat
  'ahmedabad', 'surat', 'vadodara', 'rajkot', 'bhavnagar', 'jamnagar', 'gandhinagar',
  // NCR / North India
  'noida', 'gurgaon', 'gurugram', 'faridabad', 'ghaziabad', 'meerut', 'agra',
  'lucknow', 'kanpur', 'varanasi', 'allahabad', 'prayagraj', 'jaipur', 'jodhpur',
  'chandigarh', 'ludhiana', 'amritsar', 'dehradun', 'haridwar',
  // South India
  'coimbatore', 'kochi', 'cochin', 'trivandrum', 'thiruvananthapuram', 'madurai',
  'mysore', 'mysuru', 'mangalore', 'mangaluru', 'visakhapatnam', 'vizag',
  'vijayawada', 'tirupati', 'warangal',
  // East India
  'patna', 'bhubaneswar', 'guwahati', 'ranchi',
  // MP, Chhattisgarh
  'bhopal', 'indore', 'raipur', 'jabalpur',
  // International
  'new york', 'london', 'san francisco', 'los angeles', 'chicago', 'toronto',
  'singapore', 'dubai', 'abu dhabi', 'hong kong', 'sydney', 'tokyo', 'shanghai',
  // Indian states
  'maharashtra', 'karnataka', 'tamil nadu', 'telangana', 'gujarat',
  'rajasthan', 'uttar pradesh', 'madhya pradesh', 'kerala', 'west bengal',
  'andhra pradesh', 'bihar', 'punjab', 'haryana', 'goa', 'odisha', 'assam',
  'chhattisgarh', 'jharkhand', 'uttarakhand', 'himachal pradesh',
  'india', 'usa', 'uk', 'uae', 'canada', 'australia', 'germany', 'france',
];

// Phone label patterns
const MOBILE_LABELS = /(?:mob(?:ile)?|cell|m\s*[:.]|whatsapp|wa)/i;
const LANDLINE_LABELS = /(?:tel(?:ephone)?|phone|ph\s*[:.]|off(?:ice)?|fax|t\s*[:.]|o\s*[:.]|land(?:line)?|res(?:idence)?|r\s*[:.])(?!\w)/i;

// Name prefixes — honorifics that hint a line is a person name
const NAME_PREFIXES = /^(?:mr\.?|mrs\.?|ms\.?|miss|dr\.?|prof\.?|shri\.?|smt\.?|ca\.?|cs\.?|adv\.?|er\.?|ar\.?)\s+/i;

// ============================================================
// OCR TEXT PREPROCESSING
// ============================================================

function preprocessOcrText(rawText: string): string {
  let text = rawText;

  // Fix common OCR misreads
  text = text.replace(/[''`]/g, "'");
  text = text.replace(/[""]/g, '"');
  text = text.replace(/—|–/g, '-');
  text = text.replace(/\u00a0/g, ' '); // non-breaking space

  // Fix 'l' misread as '|' or 'I' in context
  // Fix '0' misread as 'O' in phone numbers — handled per-field

  // Remove stray special characters that are OCR noise
  text = text.replace(/[^\S\n]+/g, ' '); // collapse whitespace (except newlines)

  // Remove lines that are just noise characters
  text = text.split('\n').map(line => {
    const cleaned = line.trim();
    // Remove lines that are mostly special chars (symbols, bars, dots)
    const alphanumCount = (cleaned.match(/[a-zA-Z0-9]/g) || []).length;
    if (cleaned.length > 0 && alphanumCount / cleaned.length < 0.3) {
      return ''; // likely OCR noise
    }
    return cleaned;
  }).join('\n');

  return text;
}

function cleanLine(line: string): string {
  return line
    .replace(/^[\s|:•·*\-–—=]+/, '') // leading noise
    .replace(/[\s|:•·*\-–—=]+$/, '') // trailing noise
    .replace(/\s{2,}/g, ' ')         // collapse spaces
    .trim();
}

// ============================================================
// FIELD DETECTION FUNCTIONS (with scoring)
// ============================================================

function getPhoneScore(line: string): { score: number; numbers: string[] } {
  const digits = line.replace(/\D/g, '');
  if (digits.length < 7) return { score: 0, numbers: [] };

  // Don't match lines that are primarily text with an incidental number
  const textRatio = (line.match(/[a-zA-Z]/g) || []).length / line.length;

  const numbers: string[] = [];
  PHONE_REGEX.lastIndex = 0;
  let match;
  while ((match = PHONE_REGEX.exec(line)) !== null) {
    const matchDigits = match[0].replace(/\D/g, '');
    if (matchDigits.length >= 7 && matchDigits.length <= 15) {
      numbers.push(match[0].trim());
    }
  }

  if (numbers.length === 0) return { score: 0, numbers: [] };

  let score = 5;
  // Boost: has phone labels
  if (/(?:ph(?:one)?|tel|mob(?:ile)?|cell|fax|office|off|land|res|whatsapp|contact)/i.test(line)) score += 3;
  // Boost: has + prefix (international)
  if (/\+\d/.test(line)) score += 2;
  // Boost: all digits in a well-formatted number
  if (textRatio < 0.3) score += 2;
  // Reduce: too many alpha characters (might be a misread)
  if (textRatio > 0.5) score -= 3;

  return { score: Math.max(0, score), numbers };
}

function getEmailScore(line: string): { score: number; email: string } {
  EMAIL_REGEX.lastIndex = 0;
  const match = line.match(EMAIL_REGEX);
  if (!match) return { score: 0, email: '' };

  let score = 8; // Emails are highly distinctive
  if (/(?:email|e[\-\s]?mail|mail)\s*[:.]?\s*/i.test(line)) score += 2;

  return { score, email: match[0].toLowerCase() };
}

function getWebsiteScore(line: string): { score: number; url: string } {
  if (line.includes('@')) return { score: 0, url: '' }; // It's an email, not a website

  WEBSITE_REGEX.lastIndex = 0;
  const match = line.match(WEBSITE_REGEX);
  if (!match) return { score: 0, url: '' };

  // Filter out obvious non-websites
  const url = match[0];
  if (/\.(jpg|jpeg|png|gif|pdf|doc)$/i.test(url)) return { score: 0, url: '' };

  let score = 7;
  if (/(?:www\.|https?:\/\/)/i.test(url)) score += 3;
  if (/(?:web(?:site)?|url|www|visit)\s*[:.]?\s*/i.test(line)) score += 2;

  return { score, url: url.startsWith('http') ? url : 'https://' + url };
}

function getTitleScore(line: string): number {
  let maxScore = 0;
  for (const { pattern, weight } of JOB_TITLE_PATTERNS) {
    if (pattern.test(line)) {
      maxScore = Math.max(maxScore, weight);
    }
  }

  // Boost: short line (titles are usually concise)
  const wordCount = line.split(/\s+/).length;
  if (wordCount <= 5 && maxScore > 0) maxScore += 1;

  // Reduce: has too many digits (probably not a title)
  if (/\d{3,}/.test(line) && maxScore > 0) maxScore -= 3;

  return Math.max(0, maxScore);
}

function getCompanyScore(line: string): number {
  let totalScore = 0;
  for (const { pattern, weight } of COMPANY_PATTERNS) {
    if (pattern.test(line)) {
      totalScore = Math.max(totalScore, weight);
    }
  }

  // Reduce: has @ sign (it's an email)
  if (line.includes('@')) totalScore = 0;
  // Reduce: too many digits (probably phone or address)
  const digits = (line.match(/\d/g) || []).length;
  if (digits > 5) totalScore -= 3;

  return Math.max(0, totalScore);
}

function getAddressScore(line: string): number {
  const lower = line.toLowerCase();
  let score = 0;

  // Check address keywords
  for (const kw of ADDRESS_KEYWORDS) {
    const regex = new RegExp(`\\b${kw}\\b`, 'i');
    if (regex.test(lower)) {
      score += 3;
    }
  }

  // Check city/state names
  for (const city of CITY_NAMES) {
    if (lower.includes(city)) {
      score += 4;
      break; // One city is enough
    }
  }

  // Check for pin/zip code
  if (PIN_CODE_REGEX.test(line)) score += 4;

  // Check for comma-separated segments (common in addresses)
  const commaSegments = line.split(',').length;
  if (commaSegments >= 2 && score > 0) score += 2;

  // Reduce: looks like email or website
  if (line.includes('@') || /(?:www\.|https?:\/\/)/i.test(line)) score = 0;

  return score;
}

function getNameScore(line: string, lineIndex: number, totalLines: number): number {
  const wordCount = line.split(/\s+/).length;
  const hasDigits = /\d/.test(line);
  const lineLength = line.length;

  // Immediate disqualifiers
  if (hasDigits && (line.match(/\d/g) || []).length > 2) return 0;
  if (lineLength < 2 || lineLength > 55) return 0;
  if (wordCount > 5) return 0;
  if (line.includes('@')) return 0;
  if (/(?:www\.|https?:\/\/)/.test(line)) return 0;

  let score = 5;

  // Boost: has name prefix / honorific
  if (NAME_PREFIXES.test(line)) score += 6;

  // Boost: 2-4 words (Indian names often have 3 parts)
  if (wordCount >= 2 && wordCount <= 4) score += 3;

  // Boost: appears early in the card (names are usually at the top)
  if (lineIndex === 0) score += 6;
  else if (lineIndex === 1) score += 4;
  else if (lineIndex <= 3) score += 2;
  else score -= 2;

  const words = line.split(/\s+/);

  // Boost: Title-Case (all words start uppercase)
  if (words.every(w => /^[A-Z][a-z]/.test(w))) score += 4;

  // Boost: ALL-CAPS name (very common on Indian business cards)
  // e.g. "RAJESH KUMAR SHARMA" — each word 2+ chars, no digits
  const isAllCaps = line === line.toUpperCase() && /[A-Z]{2,}/.test(line);
  if (isAllCaps && words.every(w => w.length >= 2)) score += 5;

  // Reduce: contains special characters common in non-name fields
  if (/[/:;#@{}()[\]]/.test(line)) score -= 4;

  // Reduce: contains common non-name keywords
  if (/\b(?:road|street|floor|building|pvt|ltd|llp|email|phone|tel|fax|mob|cell|web|www|extn|near|opp|landmark|gstin|gst|pan|udyam|cin)\b/i.test(line)) score -= 8;

  // Reduce: looks like a phone number pattern
  if (/^[\d\s\+\-\.\(\)]{7,}$/.test(line)) score -= 8;

  return Math.max(0, score);
}

// ============================================================
// PHONE CLASSIFICATION
// ============================================================

function classifyPhone(contextLine: string, number: string): 'mobile' | 'landline' {
  // Check explicit labels in the surrounding text
  if (MOBILE_LABELS.test(contextLine)) return 'mobile';
  if (LANDLINE_LABELS.test(contextLine)) return 'landline';

  const digits = number.replace(/\D/g, '');

  // Strip country code
  let localDigits = digits;
  if (digits.startsWith('91') && digits.length >= 12) {
    localDigits = digits.slice(2);
  } else if (digits.startsWith('0') && digits.length >= 11) {
    localDigits = digits.slice(1);
  } else if (digits.startsWith('1') && digits.length === 11) {
    localDigits = digits.slice(1);
  } else if (digits.startsWith('44') && digits.length >= 12) {
    localDigits = digits.slice(2);
  }

  // Indian mobile: 10 digits starting with 6-9
  if (localDigits.length === 10 && /^[6-9]/.test(localDigits)) {
    return 'mobile';
  }

  // Indian landline: typically 10-11 digits starting with area code (2-8)
  // Pattern: STD code (2-4 digits) + number (6-8 digits)
  if (localDigits.length >= 8 && localDigits.length <= 11 && /^[0-8]/.test(localDigits)) {
    // If it's exactly 10 digits starting with 2-5, it's likely a landline with STD
    if (localDigits.length <= 8 || /^[2-5]/.test(localDigits)) {
      return 'landline';
    }
  }

  // International with + prefix — usually mobile
  if (number.includes('+')) return 'mobile';

  // US/Canada: 10 digits → usually mobile
  if (localDigits.length === 10) return 'mobile';

  // Short numbers → landline
  if (localDigits.length <= 8) return 'landline';

  return 'mobile'; // default
}

// ============================================================
// MAIN PARSER
// ============================================================

export function parseContactFromText(rawText: string): ContactInfo {
  const processed = preprocessOcrText(rawText);
  const lines = processed
    .split('\n')
    .map(cleanLine)
    .filter(line => line.length > 1); // skip empty/single-char lines

  const contact: ContactInfo = {
    firstName: '',
    lastName: '',
    title: '',
    company: '',
    landlines: [],
    mobiles: [],
    email: '',
    website: '',
    address: '',
    notes: '',
  };

  if (lines.length === 0) return contact;

  // ---- Score every line for every possible field type ----
  interface LineAnalysis {
    line: string;
    index: number;
    phoneScore: number;
    phoneNumbers: string[];
    emailScore: number;
    emailValue: string;
    websiteScore: number;
    websiteValue: string;
    titleScore: number;
    companyScore: number;
    addressScore: number;
    nameScore: number;
    assignedAs: string | null;
  }

  const analyses: LineAnalysis[] = lines.map((line, index) => {
    const phone = getPhoneScore(line);
    const email = getEmailScore(line);
    const website = getWebsiteScore(line);

    return {
      line,
      index,
      phoneScore: phone.score,
      phoneNumbers: phone.numbers,
      emailScore: email.score,
      emailValue: email.email,
      websiteScore: website.score,
      websiteValue: website.url,
      titleScore: getTitleScore(line),
      companyScore: getCompanyScore(line),
      addressScore: getAddressScore(line),
      nameScore: getNameScore(line, index, lines.length),
      assignedAs: null,
    };
  });

  // ---- Pass 1: Assign high-confidence fields (email, website, phone) ----
  for (const a of analyses) {
    // Email — very distinctive
    if (a.emailScore >= 8 && !contact.email) {
      contact.email = a.emailValue;
      a.assignedAs = 'email';
    }

    // Website — very distinctive
    if (a.websiteScore >= 7 && !contact.website && a.assignedAs !== 'email') {
      contact.website = a.websiteValue;
      a.assignedAs = 'website';
    }

    // Phone numbers
    if (a.phoneScore >= 5 && a.assignedAs === null) {
      for (const num of a.phoneNumbers) {
        const type = classifyPhone(a.line, num);
        if (type === 'mobile') {
          contact.mobiles.push(num);
        } else {
          contact.landlines.push(num);
        }
      }
      a.assignedAs = 'phone';
    }
  }

  // ---- Pass 2: Assign company and title (need to disambiguate) ----
  // Find best company line
  let bestCompanyIdx = -1;
  let bestCompanyScore = 0;
  for (const a of analyses) {
    if (a.assignedAs !== null) continue;
    if (a.companyScore > bestCompanyScore) {
      bestCompanyScore = a.companyScore;
      bestCompanyIdx = a.index;
    }
  }
  if (bestCompanyIdx >= 0 && bestCompanyScore >= 5) {
    const a = analyses.find(x => x.index === bestCompanyIdx)!;
    contact.company = a.line;
    a.assignedAs = 'company';
  }

  // Find best title line (but not the same as company)
  let bestTitleIdx = -1;
  let bestTitleScore = 0;
  for (const a of analyses) {
    if (a.assignedAs !== null) continue;
    if (a.titleScore > bestTitleScore) {
      bestTitleScore = a.titleScore;
      bestTitleIdx = a.index;
    }
  }
  if (bestTitleIdx >= 0 && bestTitleScore >= 4) {
    const a = analyses.find(x => x.index === bestTitleIdx)!;
    contact.title = a.line;
    a.assignedAs = 'title';
  }

  // ---- Pass 3: Address lines ----
  const addressParts: string[] = [];
  for (const a of analyses) {
    if (a.assignedAs !== null) continue;
    if (a.addressScore >= 4) {
      addressParts.push(a.line);
      a.assignedAs = 'address';
    }
  }
  // Also grab adjacent unassigned lines near address lines (address continuation)
  for (let i = 0; i < analyses.length; i++) {
    const a = analyses[i];
    if (a.assignedAs === 'address') {
      // Check next line — if it's unassigned and could be address continuation
      const next = analyses[i + 1];
      if (next && next.assignedAs === null && next.addressScore >= 2) {
        addressParts.push(next.line);
        next.assignedAs = 'address';
      }
    }
  }
  if (addressParts.length > 0) {
    contact.address = addressParts.join(', ');
  }

  // ---- Pass 4: Name detection ----
  // Strategy: Find the best name candidate from remaining unassigned lines
  // Use email username as a hint if available
  let emailNameHint = '';
  if (contact.email) {
    const username = contact.email.split('@')[0];
    // Common email patterns: first.last, firstlast, first_last
    emailNameHint = username.replace(/[._\-\d]/g, ' ').trim().toLowerCase();
  }

  let bestNameIdx = -1;
  let bestNameScore = 0;
  for (const a of analyses) {
    if (a.assignedAs !== null) continue;
    let score = a.nameScore;

    // Boost: if line content matches email username pattern
    if (emailNameHint && emailNameHint.length > 2) {
      const lineLower = a.line.toLowerCase();
      const hintWords = emailNameHint.split(/\s+/);
      for (const hw of hintWords) {
        if (hw.length > 2 && lineLower.includes(hw)) {
          score += 4;
          break;
        }
      }
    }

    if (score > bestNameScore) {
      bestNameScore = score;
      bestNameIdx = a.index;
    }
  }

  if (bestNameIdx >= 0 && bestNameScore >= 3) {
    const a = analyses.find(x => x.index === bestNameIdx)!;
    let nameLine = a.line;

    // Remove prefix for parsing but keep for display
    nameLine = nameLine.replace(NAME_PREFIXES, '').trim();

    const nameParts = nameLine.split(/\s+/);
    if (nameParts.length >= 3) {
      contact.firstName = nameParts[0];
      contact.lastName = nameParts.slice(1).join(' ');
    } else if (nameParts.length === 2) {
      contact.firstName = nameParts[0];
      contact.lastName = nameParts[1];
    } else if (nameParts.length === 1) {
      contact.firstName = nameParts[0];
    }
    a.assignedAs = 'name';
  }

  // ---- Pass 5: Second chance — if no company found, check if any remaining line could be ----
  if (!contact.company) {
    for (const a of analyses) {
      if (a.assignedAs !== null) continue;
      if (a.companyScore >= 3) {
        contact.company = a.line;
        a.assignedAs = 'company';
        break;
      }
    }
  }

  // ---- Pass 6: If no title found, look for a line near the name that might be a title ----
  if (!contact.title && bestNameIdx >= 0) {
    // Title often appears right after the name
    for (let offset of [1, -1, 2]) {
      const idx = bestNameIdx + offset;
      if (idx >= 0 && idx < analyses.length) {
        const a = analyses[idx];
        if (a.assignedAs === null && a.titleScore >= 2) {
          contact.title = a.line;
          a.assignedAs = 'title';
          break;
        }
      }
    }
  }

  // ---- Remaining unassigned lines → notes ----
  const noteLines: string[] = [];
  for (const a of analyses) {
    if (a.assignedAs === null && a.line.length > 2) {
      noteLines.push(a.line);
    }
  }

  // Also scan ALL text for GST/PAN/Udyam that may be embedded in any line
  const allText = lines.join(' ');
  const gstMatches = [...allText.matchAll(GST_REGEX)].map(m => `GSTIN: ${m[0]}`);
  const panMatches = [...allText.matchAll(PAN_REGEX)]
    .map(m => m[0])
    .filter(pan => !allText.match(new RegExp(`GSTIN.*${pan}`))) // avoid duplicating GST PAN
    .map(p => `PAN: ${p}`);
  const udyamMatches = [...allText.matchAll(UDYAM_REGEX)].map(m => `Udyam: ${m[0]}`);

  const specialIds = [...gstMatches, ...panMatches, ...udyamMatches];
  const allNotes = [...noteLines, ...specialIds];
  if (allNotes.length > 0) {
    contact.notes = [...new Set(allNotes)].join(' | ');
  }

  // ---- Dedup phone numbers ----
  contact.mobiles = [...new Set(contact.mobiles)];
  contact.landlines = [...new Set(contact.landlines)];

  return contact;
}

// ============================================================
// PARTIAL PARSER (for live preview during OCR)
// ============================================================

export function parsePartialContact(rawText: string): Partial<ContactInfo> {
  const partial: Partial<ContactInfo> = {};
  const processed = preprocessOcrText(rawText);

  // Email
  EMAIL_REGEX.lastIndex = 0;
  const emailMatch = processed.match(EMAIL_REGEX);
  if (emailMatch) partial.email = emailMatch[0].toLowerCase();

  // Phones
  PHONE_REGEX.lastIndex = 0;
  const phoneMatches = processed.match(PHONE_REGEX) || [];
  const mobiles: string[] = [];
  const landlines: string[] = [];
  for (const m of phoneMatches) {
    const digits = m.replace(/\D/g, '');
    if (digits.length >= 7 && digits.length <= 15) {
      const type = classifyPhone(processed, m.trim());
      if (type === 'mobile') mobiles.push(m.trim());
      else landlines.push(m.trim());
    }
  }
  if (mobiles.length > 0) partial.mobiles = [...new Set(mobiles)];
  if (landlines.length > 0) partial.landlines = [...new Set(landlines)];

  // Website
  WEBSITE_REGEX.lastIndex = 0;
  const webMatch = processed.match(WEBSITE_REGEX);
  if (webMatch) {
    const url = webMatch.find(m => !m.includes('@'));
    if (url) partial.website = url.startsWith('http') ? url : 'https://' + url;
  }

  // Quick name attempt from first few lines
  const lines = processed.split('\n').map(cleanLine).filter(l => l.length > 2);
  for (const line of lines.slice(0, 3)) {
    const words = line.split(/\s+/);
    const hasDigits = /\d/.test(line);
    const isEmail = line.includes('@');
    if (!hasDigits && !isEmail && words.length >= 2 && words.length <= 4 && line.length <= 40) {
      const cleaned = line.replace(NAME_PREFIXES, '').trim();
      const parts = cleaned.split(/\s+/);
      partial.firstName = parts[0];
      partial.lastName = parts.slice(1).join(' ');
      break;
    }
  }

  return partial;
}
