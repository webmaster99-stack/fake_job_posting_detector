// Turn the API's explanation into readable labels, and find suspicious terms in the posting text.
// Feature names come from fjd.features.build (TABULAR_FEATURES, ONE_HOT_COLUMNS).

import type { Signal } from "./api";

const FIELD_NAMES: Record<string, string> = {
  title: "title",
  company_profile: "company profile",
  description: "description",
  requirements: "requirements",
  benefits: "benefits",
  location: "location",
  department: "department",
  salary_range: "salary range",
  employment_type: "employment type",
  required_experience: "required experience",
  required_education: "required education",
  industry: "industry",
  function: "job function",
  country: "country",
};

const field = (name: string) => FIELD_NAMES[name] ?? name.replaceAll("_", " ");

// The featurizer replaces links, emails and phone numbers with these terms.
const PLACEHOLDER_TERMS: Record<string, string> = {
  xxurl: "a link",
  xxemail: "an email address",
  xxphone: "a phone number",
};

const TABULAR_LABELS: Record<string, string> = {
  telecommuting: "Remote job",
  has_company_logo: "Company logo shown",
  has_questions: "Screening questions",
  has_url: "Contains a link",
  has_email: "Contains an email address",
  has_phone: "Contains a phone number",
};

export function label(signal: Signal): string {
  const { kind, feature } = signal;
  if (kind === "text") {
    const placeholder = PLACEHOLDER_TERMS[feature];
    return placeholder ? `Mentions ${placeholder}` : `“${feature}”`;
  }
  if (kind === "categorical") {
    const [column, value] = feature.split("=", 2);
    if (value === "(missing)") return `No ${field(column)} given`;
    if (value?.startsWith("infrequent")) return `Uncommon ${field(column)}`;
    return `${capitalize(field(column))}: ${value}`;
  }
  if (feature.startsWith("missing_")) return `${capitalize(field(feature.slice(8)))} left empty`;
  if (feature.startsWith("log_words_")) return `Length of the ${field(feature.slice(10))}`;
  return TABULAR_LABELS[feature] ?? capitalize(feature.replaceAll("_", " "));
}

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

// What the backend replaces with placeholders before scoring (fjd.serving.anonymize), in the
// same order: emails, then links, then phone numbers. The model never sees the words inside them.
const PLACEHOLDER_PATTERNS: [term: string, pattern: string][] = [
  ["xxemail", String.raw`[\w.+-]+@[\w-]+(?:\.[\w-]+)+`],
  ["xxurl", String.raw`(?:https?://|www\.)[^\s<>"']+`],
  ["xxphone", String.raw`(?<![\w#])\+?\(?\d[\d\s().-]{6,}\d(?!\w)`],
];
const PHONE_DIGITS = { min: 9, max: 15 };
const CONTACT_FEATURES = new Set(["has_url", "has_email", "has_phone"]);

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// A TF-IDF term is one or two lowercased words of 2+ characters (sklearn's default tokenizer);
// in the original text the words may be separated by any non-word characters.
const termPattern = (term: string) =>
  String.raw`\b${term.split(" ").map(escape).join(String.raw`[\W_]+`)}\b`;

export type Segment = { text: string; mark?: "fraud" | "real" };

/** Split `text` into plain and marked segments for the explanation's text terms. */
export function highlight(text: string, signals: Signal[]): Segment[] {
  // The tabular has_url/has_email/has_phone features mark the same spans as their placeholder terms.
  const spanTerm = (s: Signal) =>
    s.kind === "tabular" && CONTACT_FEATURES.has(s.feature) ? `xx${s.feature.slice(4)}` : null;
  const direction = new Map(
    signals
      .filter((s) => s.contribution !== 0 && (s.kind === "text" || spanTerm(s)))
      .map((s) => [spanTerm(s) ?? s.feature, s.contribution > 0]),
  );
  if (!text || direction.size === 0) return [{ text }];
  // Placeholders come first, so their spans are consumed whole; then longer terms first, so a
  // bigram wins over the unigram inside it.
  const placeholders = new Set(PLACEHOLDER_PATTERNS.map(([term]) => term));
  const words = [...direction.keys()]
    .filter((t) => !placeholders.has(t))
    .sort((a, b) => b.length - a.length);
  const alternatives: [term: string, pattern: string][] = [
    ...PLACEHOLDER_PATTERNS,
    ...words.map((t): [string, string] => [t, termPattern(t)]),
  ];
  const regex = new RegExp(alternatives.map(([, p]) => `(${p})`).join("|"), "giu");

  const segments: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(regex)) {
    const index = match.index ?? 0;
    const term = alternatives[match.slice(1).findIndex((g) => g !== undefined)][0];
    const digits = match[0].replace(/\D/g, "").length;
    if (term === "xxphone" && (digits < PHONE_DIGITS.min || digits > PHONE_DIGITS.max)) continue;
    const fraud = direction.get(term);
    if (fraud === undefined) continue; // a link etc. that isn't in the explanation: plain text
    if (index > last) segments.push({ text: text.slice(last, index) });
    segments.push({ text: match[0], mark: fraud ? "fraud" : "real" });
    last = index + match[0].length;
  }
  if (last < text.length) segments.push({ text: text.slice(last) });
  return segments;
}
