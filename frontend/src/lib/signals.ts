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

// What the backend replaces with placeholders (fjd.serving), roughly.
const PLACEHOLDER_PATTERNS: Record<string, string> = {
  xxurl: String.raw`(?:https?://|www\.)[^\s<>"']+`,
  xxemail: String.raw`[\w.+-]+@[\w-]+(?:\.[\w-]+)+`,
  xxphone: String.raw`\+?\(?\d[\d\s().-]{6,}\d`,
};

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// A TF-IDF term is one or two lowercased words of 2+ characters (sklearn's default tokenizer);
// in the original text the words may be separated by any non-word characters.
function termPattern(term: string): string {
  const placeholder = PLACEHOLDER_PATTERNS[term];
  if (placeholder) return placeholder;
  return String.raw`\b${term.split(" ").map(escape).join(String.raw`[\W_]+`)}\b`;
}

export type Segment = { text: string; mark?: "fraud" | "real" };

/** Split `text` into plain and marked segments for the explanation's text terms. */
export function highlight(text: string, signals: Signal[]): Segment[] {
  const terms = signals.filter((s) => s.kind === "text" && s.contribution !== 0);
  if (!text || terms.length === 0) return [{ text }];
  // Longer terms first, so a bigram wins over the unigram inside it.
  terms.sort((a, b) => b.feature.length - a.feature.length);
  const direction = new Map(terms.map((s) => [s.feature, s.contribution > 0]));
  const regex = new RegExp(terms.map((s) => `(${termPattern(s.feature)})`).join("|"), "giu");

  const segments: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(regex)) {
    const index = match.index ?? 0;
    if (match[0].length === 0) continue;
    const group = match.slice(1).findIndex((g) => g !== undefined);
    const fraud = direction.get(terms[group].feature);
    if (index > last) segments.push({ text: text.slice(last, index) });
    segments.push({ text: match[0], mark: fraud ? "fraud" : "real" });
    last = index + match[0].length;
  }
  if (last < text.length) segments.push({ text: text.slice(last) });
  return segments;
}
