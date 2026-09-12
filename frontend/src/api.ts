export type CarFeatures = {
  make: string;
  model: string;
  year: number;
  mileage: number;
  condition: string;
  fuel_type: string;
  transmission: string;
  body_type: string;
  accident_history: boolean;
  owners: number;
};

export type PredictionResponse = {
  predicted_price: number;
  price_low: number;
  price_high: number;
  currency: string;
  confidence: number;
  advice: string;
  model_version: number;
  feature_importance: Record<string, number>;
};

export type ModelInfo = {
  model_version: number;
  samples_used: number;
  metrics: { mae?: number; rmse?: number; r2?: number };
  makes: string[];
  models_by_make: Record<string, string[]>;
  conditions: string[];
  fuel_types: string[];
  transmissions: string[];
  body_types: string[];
  last_trained_at: string | null;
  feature_names: string[];
  status: string;
};

export type FeedbackResponse = {
  message: string;
  model_version: number;
  samples_used: number;
  metrics: { mae?: number; rmse?: number; r2?: number };
  previous_metrics: { mae?: number; rmse?: number; r2?: number };
};

export type HistoryItem = {
  id: number;
  timestamp: string;
  make: string;
  model: string;
  year: number;
  predicted_price: number | null;
  actual_price: number;
  notes: string | null;
};

const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  info: () => request<ModelInfo>("/model/info"),
  predict: (features: CarFeatures) =>
    request<PredictionResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(features),
    }),
  feedback: (features: CarFeatures, actual_price: number, notes?: string) =>
    request<FeedbackResponse>("/feedback", {
      method: "POST",
      body: JSON.stringify({ features, actual_price, notes: notes || null }),
    }),
  train: (reset = false) =>
    request<{ message: string; model_version: number; samples_used: number; metrics: ModelInfo["metrics"] }>(
      `/train?reset_dataset=${reset}`,
      { method: "POST" },
    ),
  history: (limit = 12) =>
    request<{ items: HistoryItem[]; total: number }>(`/history?limit=${limit}`),
};
