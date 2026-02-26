/**
 * Contact Parser - Extracts structured contact information from raw OCR text.
 * Uses regex patterns and heuristics to identify fields from business card text.
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

// --- Regex Patterns ---

const EMAIL_REGEX = /[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/gi;

const PHONE_REGEX = /(?:[\+]?\d{1,3}[\s\-.]?)?\(?\d{2,5}\)?[\s\-.]?\d{2,5}[\s\-.]?\d{2,6}/g;

const WEBSITE_REGEX = /(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9\-]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:\/[^\s]*)*/gi;

const PIN_CODE_REGEX = /\b\d{5,6}\b/;

// Common job title keywords (case-insensitive matching)
const JOB_TITLE_KEYWORDS = [
  'ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio', 'vp',
  'president', 'vice president',
  'director', 'manager', 'head', 'lead', 'chief',
  'engineer', 'developer', 'architect', 'designer', 'analyst',
  'consultant', 'advisor', 'specialist', 'coordinator',
  'executive', 'officer', 'partner', 'founder', 'co-founder',
  'associate', 'assistant', 'supervisor', 'administrator',
  'accountant', 'advocate', 'attorney', 'lawyer',
  'doctor', 'dr\\.', 'professor', 'prof\\.',
  'sales', 'marketing', 'operations', 'finance',
  'senior', 'junior', 'sr\\.', 'jr\\.',
  'general manager', 'managing director', 'proprietor', 'owner',
];

// Common company suffixes
const COMPANY_SUFFIXES = [
  'ltd', 'limited', 'llc', 'llp', 'inc', 'incorporated',
  'corp', 'corporation', 'co\\.', 'company',
  'pvt', 'private', 'group', 'enterprise', 'enterprises',
  'solutions', 'services', 'technologies', 'tech',
  'industries', 'associates', 'consultants', 'consulting',
  'international', 'global', 'ventures', 'holdings',
  'foundation', 'institute', 'academy', 'labs', 'studio',
  'systems', 'infotech', 'infosystems',
];

// Phone label patterns that indicate mobile
const MOBILE_LABELS = /(?:mob(?:ile)?|cell|m\s*:|whatsapp)/i;
const LANDLINE_LABELS = /(?:tel(?:ephone)?|phone|ph|off(?:ice)?|fax|t\s*:|o\s*:|land(?:line)?)/i;

// --- Helper Functions ---

function cleanLine(line: string): string {
  return line.replace(/[|]/g, '').trim();
}

function isPhoneLine(line: string): boolean {
  const digits = line.replace(/\D/g, '');
  return digits.length >= 7;
}

function isEmailLine(line: string): boolean {
  return EMAIL_REGEX.test(line);
}

function isWebsiteLine(line: string): boolean {
  // Reset regex
  WEBSITE_REGEX.lastIndex = 0;
  const match = WEBSITE_REGEX.exec(line);
  if (!match) return false;
  // Exclude emails matched as websites
  if (line.includes('@')) return false;
  return true;
}

function isTitleLine(line: string): boolean {
  const lower = line.toLowerCase();
  return JOB_TITLE_KEYWORDS.some(keyword => {
    const regex = new RegExp(`\\b${keyword}\\b`, 'i');
    return regex.test(lower);
  });
}

function isCompanyLine(line: string): boolean {
  const lower = line.toLowerCase();
  return COMPANY_SUFFIXES.some(suffix => {
    const regex = new RegExp(`\\b${suffix}\\b`, 'i');
    return regex.test(lower);
  });
}

function isAddressLine(line: string): boolean {
  const lower = line.toLowerCase();
  // Common address indicators
  const addressKeywords = [
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln',
    'building', 'bldg', 'floor', 'suite', 'ste', 'flat', 'plot',
    'sector', 'block', 'nagar', 'colony', 'area', 'park',
    'city', 'town', 'district', 'state', 'country',
    'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'pune',
    'hyderabad', 'ahmedabad', 'india', 'usa',
    'near', 'opp', 'opposite', 'beside', 'behind',
  ];
  const hasAddressKeyword = addressKeywords.some(kw => lower.includes(kw));
  const hasPinCode = PIN_CODE_REGEX.test(line);
  return hasAddressKeyword || hasPinCode;
}

function classifyPhone(line: string, number: string): 'mobile' | 'landline' {
  // Check if line has explicit labels
  if (MOBILE_LABELS.test(line)) return 'mobile';
  if (LANDLINE_LABELS.test(line)) return 'landline';

  // Heuristic: Indian mobile numbers start with 6-9 and have 10 digits
  const digits = number.replace(/\D/g, '');
  
  // Remove country code if present
  let localDigits = digits;
  if (digits.startsWith('91') && digits.length > 10) {
    localDigits = digits.slice(2);
  } else if (digits.startsWith('1') && digits.length === 11) {
    localDigits = digits.slice(1);
  }

  // 10-digit number starting with 6-9 is likely mobile (Indian)
  if (localDigits.length === 10 && /^[6-9]/.test(localDigits)) {
    return 'mobile';
  }

  // Numbers with +91 prefix usually mobile
  if (number.includes('+91') || number.includes('+1')) {
    return 'mobile';
  }

  // Shorter numbers are usually landlines
  if (localDigits.length <= 8) {
    return 'landline';
  }

  // Default to mobile for 10+ digit numbers
  return localDigits.length >= 10 ? 'mobile' : 'landline';
}

function extractPhoneNumbers(text: string): string[] {
  const phones: string[] = [];
  const matches = text.match(PHONE_REGEX) || [];
  
  for (const match of matches) {
    const digits = match.replace(/\D/g, '');
    // Only include numbers with 7+ digits (skip short numbers like dates/years)
    if (digits.length >= 7 && digits.length <= 15) {
      const cleaned = match.trim();
      if (!phones.includes(cleaned)) {
        phones.push(cleaned);
      }
    }
  }
  
  return phones;
}

// --- Main Parser ---

export function parseContactFromText(rawText: string): ContactInfo {
  const lines = rawText
    .split('\n')
    .map(cleanLine)
    .filter(line => line.length > 0);

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

  // Extract emails
  EMAIL_REGEX.lastIndex = 0;
  const emailMatches = rawText.match(EMAIL_REGEX);
  if (emailMatches && emailMatches.length > 0) {
    contact.email = emailMatches[0].toLowerCase();
  }

  // Extract websites (exclude email domains)
  const websiteMatches: string[] = [];
  for (const line of lines) {
    WEBSITE_REGEX.lastIndex = 0;
    const matches = line.match(WEBSITE_REGEX) || [];
    for (const m of matches) {
      if (!m.includes('@') && !emailMatches?.some(e => m.includes(e))) {
        websiteMatches.push(m);
      }
    }
  }
  if (websiteMatches.length > 0) {
    let url = websiteMatches[0];
    if (!url.startsWith('http')) {
      url = 'https://' + url;
    }
    contact.website = url;
  }

  // Classify each line
  const usedLines = new Set<number>();
  const phoneLines: number[] = [];
  const addressLines: string[] = [];
  let titleLine = '';
  let companyLine = '';

  // First pass: Extract phones, emails, websites (definite matches)
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Phone numbers
    if (isPhoneLine(line) && !isEmailLine(line) && !isWebsiteLine(line)) {
      const phones = extractPhoneNumbers(line);
      for (const phone of phones) {
        const type = classifyPhone(line, phone);
        if (type === 'mobile') {
          contact.mobiles.push(phone);
        } else {
          contact.landlines.push(phone);
        }
      }
      phoneLines.push(i);
      usedLines.add(i);
    }

    // Email line (already extracted above, just mark)
    if (isEmailLine(line)) {
      usedLines.add(i);
    }

    // Website line
    if (isWebsiteLine(line) && !isEmailLine(line)) {
      usedLines.add(i);
    }
  }

  // Second pass: Identify title, company, address from remaining lines
  for (let i = 0; i < lines.length; i++) {
    if (usedLines.has(i)) continue;
    const line = lines[i];

    if (!companyLine && isCompanyLine(line)) {
      companyLine = line;
      contact.company = line;
      usedLines.add(i);
    } else if (!titleLine && isTitleLine(line)) {
      titleLine = line;
      contact.title = line;
      usedLines.add(i);
    } else if (isAddressLine(line)) {
      addressLines.push(line);
      usedLines.add(i);
    }
  }

  // Combine address lines
  if (addressLines.length > 0) {
    contact.address = addressLines.join(', ');
  }

  // Third pass: Name detection
  // The name is typically the most prominent text (first unused line or first line)
  // Heuristic: first 1-3 lines that aren't phone/email/website/title/company/address
  const candidateNameLines: string[] = [];
  for (let i = 0; i < Math.min(lines.length, 5); i++) {
    if (usedLines.has(i)) continue;
    const line = lines[i];
    // Name lines are usually short (2-4 words) and don't contain digits
    const wordCount = line.split(/\s+/).length;
    const hasDigits = /\d/.test(line);
    if (wordCount <= 5 && !hasDigits && line.length <= 40) {
      candidateNameLines.push(line);
      usedLines.add(i);
      break; // Usually just one name line
    }
  }

  if (candidateNameLines.length > 0) {
    const nameParts = candidateNameLines[0].split(/\s+/);
    if (nameParts.length >= 2) {
      contact.firstName = nameParts[0];
      contact.lastName = nameParts.slice(1).join(' ');
    } else if (nameParts.length === 1) {
      contact.firstName = nameParts[0];
    }
  }

  // Remaining unused lines go to notes
  const noteLines: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (!usedLines.has(i)) {
      const line = lines[i];
      // Skip very short lines (probably OCR noise)
      if (line.length > 2) {
        noteLines.push(line);
      }
    }
  }
  if (noteLines.length > 0) {
    contact.notes = noteLines.join(' | ');
  }

  // Deduplicate phone numbers
  contact.mobiles = [...new Set(contact.mobiles)];
  contact.landlines = [...new Set(contact.landlines)];

  return contact;
}

/**
 * Provides incremental parsing feedback as OCR progresses.
 * Extracts whatever can be found from partial text.
 */
export function parsePartialContact(rawText: string): Partial<ContactInfo> {
  const partial: Partial<ContactInfo> = {};

  // Try to extract whatever is available
  EMAIL_REGEX.lastIndex = 0;
  const emailMatch = rawText.match(EMAIL_REGEX);
  if (emailMatch) partial.email = emailMatch[0].toLowerCase();

  const phones = extractPhoneNumbers(rawText);
  if (phones.length > 0) {
    partial.mobiles = [];
    partial.landlines = [];
    for (const phone of phones) {
      if (classifyPhone(rawText, phone) === 'mobile') {
        partial.mobiles.push(phone);
      } else {
        partial.landlines.push(phone);
      }
    }
  }

  WEBSITE_REGEX.lastIndex = 0;
  const webMatch = rawText.match(WEBSITE_REGEX);
  if (webMatch) {
    const url = webMatch.find(m => !m.includes('@'));
    if (url) partial.website = url.startsWith('http') ? url : 'https://' + url;
  }

  return partial;
}
