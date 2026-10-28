// Client for the FastAPI backend (backend/main.py). Types mirror backend/schemas.py.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

export type Posting = {
  title: string;
  company_profile?: string;
  description?: string;
  requirements?: string;
  benefits?: string;
  location?: string;
  department?: string;
  salary_range?: string;
  employment_type?: string;
  required_experience?: string;
  required_education?: string;
  industry?: string;
  function?: string;
  telecommuting: boolean;
  has_company_logo: boolean;
  has_questions: boolean;
};

export type SignalKind = "text" | "tabular" | "categorical";

export type Signal = {
  kind: SignalKind;
  feature: string; // a term, a tabular feature or "<column>=<value>"
  contribution: number; // > 0 pushes toward fraud
};

export type PredictionResult = {
  prediction_id: string | null;
  fraud: boolean;
  probability: number;
  threshold: number;
  model_name: string;
  model_version: string | null;
  explanation: Signal[];
};

export type ModelInfo = {
  model_name: string;
  registered_model: string | null;
  version: string | null;
  alias: string | null;
  threshold: number;
  explanation_method: string | null;
  metrics: Record<string, number>;
};

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const { timeoutMs = 90_000, ...rest } = init ?? {};
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...rest,
      signal: rest.signal ?? AbortSignal.timeout(timeoutMs),
    });
  } catch {
    throw new ApiError("The server didn't answer. Check your connection and try again.");
  }
  if (response.ok) return (await response.json()) as T;
  throw new ApiError(await errorMessage(response), response.status);
}

async function errorMessage(response: Response): Promise<string> {
  if (response.status === 429) {
    const retry = response.headers.get("Retry-After");
    return `Too many checks in a short time. Try again in ${retry ?? "a minute"} seconds.`;
  }
  if (response.status === 422) {
    return "Some fields are too long or invalid. Shorten them and try again.";
  }
  if (response.status === 503) {
    return "The database is unavailable right now. Try again in a minute.";
  }
  return `The server returned an error (${response.status}). Try again in a minute.`;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const health = (timeoutMs: number) =>
  request<{ status: "ok" }>("/health", { cache: "no-store", timeoutMs });

export const modelInfo = () => request<ModelInfo>("/model-info");

export const predict = (posting: Posting) =>
  request<PredictionResult>("/predict", json(posting));

export const sendFeedback = (predictionId: string, correct: boolean, comment?: string) =>
  request<{ prediction_id: string; correct: boolean }>(
    "/feedback",
    json({ prediction_id: predictionId, correct, comment: comment || null }),
  );
