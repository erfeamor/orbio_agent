// Typed client for the FastAPI service (src/agent/interfaces/api.py).

import type { components } from "./api.gen";

// Types are generated from the API's OpenAPI schema (`make api-types`), never written by hand.
type Schemas = components["schemas"];
export type Phase = Schemas["Phase"];
export type Profile = Schemas["StudentProfile"];
export type ProfileField = Schemas["ProfileField"];
export type Recommendation = Schemas["Recommendation"];
export type TurnResponse = Schemas["TurnResponse"];
export type Health = Schemas["HealthResponse"];

const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(0, `Can't reach the advisor API at ${API_URL}. Is \`make serve\` running?`);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body: unknown = await response.json();
      if (body && typeof body === "object" && "detail" in body && typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // Not JSON: keep the status text.
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const getHealth = () => request<Health>("/health");

/** `language` (e.g. navigator.language) picks the language of the consent message. */
export const startSession = (speak: boolean, language?: string) =>
  postJson<TurnResponse>("/sessions", { speak, language });

export const sendText = (conversationId: string, text: string, speak: boolean) =>
  postJson<TurnResponse>(`/sessions/${conversationId}/messages`, { text, speak });

export function sendAudio(conversationId: string, clip: Blob): Promise<TurnResponse> {
  const extension = clip.type.includes("mp4") ? "m4a" : clip.type.includes("ogg") ? "ogg" : "webm";
  const form = new FormData();
  form.append("audio", clip, `clip.${extension}`);
  return request<TurnResponse>(`/sessions/${conversationId}/audio`, { method: "POST", body: form });
}
