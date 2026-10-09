// Typed client for the FastAPI service (src/agent/interfaces/api.py).

export type Phase = "consent" | "collecting" | "confirming" | "done";

export interface Profile {
  school_year: number | null;
  specialization: string | null;
  interests: string | null;
  magic_orientation: string | null;
  learning_modality: string | null;
  weekly_free_hours: number | null;
}

export interface Recommendation {
  name: string;
  reason: string;
}

export interface TurnResponse {
  conversation_id: string;
  transcript: string | null;
  reply: string;
  audio_base64: string | null;
  audio_media_type: string | null;
  phase: Phase;
  language: string;
  profile: Profile;
  field_errors: Record<string, string>;
  recommendations: Recommendation[];
  summary: string[] | null;
}

export interface Health {
  status: string;
  speech_to_text: boolean;
  stt_provider: string | null;
  text_to_speech: boolean;
}

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

export const startSession = (speak: boolean) => postJson<TurnResponse>("/sessions", { speak });

export const sendText = (conversationId: string, text: string, speak: boolean) =>
  postJson<TurnResponse>(`/sessions/${conversationId}/messages`, { text, speak });

export function sendAudio(conversationId: string, clip: Blob): Promise<TurnResponse> {
  const extension = clip.type.includes("mp4") ? "m4a" : clip.type.includes("ogg") ? "ogg" : "webm";
  const form = new FormData();
  form.append("audio", clip, `clip.${extension}`);
  return request<TurnResponse>(`/sessions/${conversationId}/audio`, { method: "POST", body: form });
}
