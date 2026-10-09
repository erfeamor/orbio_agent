import { type FormEvent, type PointerEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  type Health,
  type TurnResponse,
  getHealth,
  sendAudio,
  sendText,
  startSession,
} from "./api";
import { DebugPanel } from "./DebugPanel";
import { useRecorder } from "./useRecorder";

interface ChatMessage {
  role: "agent" | "student";
  text: string;
  unclear?: boolean;
}

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [turn, setTurn] = useState<TurnResponse | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [speakReplies, setSpeakReplies] = useState(true);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const recorder = useRecorder();
  // True between a press that started recording and its release.
  const pressedRef = useRef(false);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  const stopAudio = () => {
    audioRef.current?.pause();
    audioRef.current = null;
  };

  const play = useCallback(
    (response: TurnResponse) => {
      if (!speakReplies || !response.audio_base64) return;
      stopAudio();
      const audio = new Audio(`data:${response.audio_media_type};base64,${response.audio_base64}`);
      audioRef.current = audio;
      // Autoplay can be blocked until the user interacts; the text is on screen anyway.
      audio.play().catch(() => undefined);
    },
    [speakReplies],
  );

  /** Runs one API call, keeping busy/error state consistent. */
  const run = async (call: () => Promise<TurnResponse>, studentText?: (r: TurnResponse) => ChatMessage) => {
    setBusy(true);
    setError(null);
    setHint(null);
    try {
      const response = await call();
      setTurn(response);
      setMessages((current) => [
        ...current,
        ...(studentText ? [studentText(response)] : []),
        { role: "agent", text: response.reply },
      ]);
      play(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const start = () => {
    stopAudio();
    setMessages([]);
    setTurn(null);
    void run(() => startSession(speakReplies));
  };

  const submitText = (event: FormEvent) => {
    event.preventDefault();
    const text = draft.trim();
    if (!turn || !text || busy) return;
    setDraft("");
    void run(() => sendText(turn.conversation_id, text, speakReplies), () => ({ role: "student", text }));
  };

  const voiceAvailable = health?.speech_to_text ?? false;
  const canTalk = Boolean(turn) && voiceAvailable && !busy;

  const pressTalk = async () => {
    if (!canTalk || pressedRef.current) return;
    pressedRef.current = true;
    stopAudio(); // talking over the advisor interrupts it
    setHint(null);
    try {
      await recorder.start();
    } catch (err) {
      pressedRef.current = false;
      setError(err instanceof Error ? err.message : "Microphone access was denied.");
    }
  };

  const releaseTalk = async () => {
    if (!pressedRef.current) return;
    pressedRef.current = false;
    const clip = await recorder.stop();
    if (!turn) return;
    if (!clip) {
      setHint("Hold the button while you speak.");
      return;
    }
    void run(
      () => sendAudio(turn.conversation_id, clip),
      (response) =>
        response.transcript
          ? { role: "student", text: response.transcript }
          : { role: "student", text: "(couldn't make that out)", unclear: true },
    );
  };

  const onPointerDown = (event: PointerEvent<HTMLButtonElement>) => {
    // Keep receiving pointer events even if the finger slides off the button.
    event.currentTarget.setPointerCapture(event.pointerId);
    void pressTalk();
  };

  // Space bar works as push-to-talk too, unless typing in the text box.
  useEffect(() => {
    const isTyping = (event: KeyboardEvent) => event.target instanceof HTMLInputElement;
    const down = (event: KeyboardEvent) => {
      if (event.code !== "Space" || event.repeat || isTyping(event)) return;
      event.preventDefault();
      void pressTalk();
    };
    const up = (event: KeyboardEvent) => {
      if (event.code !== "Space" || isTyping(event)) return;
      event.preventDefault();
      void releaseTalk();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  });

  return (
    <div className="layout">
      <main className="chat">
        <header className="chat-header">
          <div>
            <h1>Academy Advisor</h1>
            <p className="muted">Find your clubs and advanced classes</p>
          </div>
          <div className="header-actions">
            <label className="toggle">
              <input
                type="checkbox"
                checked={speakReplies}
                onChange={(e) => setSpeakReplies(e.target.checked)}
                disabled={!health?.text_to_speech}
              />
              Speak replies
            </label>
            <button onClick={start} disabled={busy || !health}>
              {turn ? "New conversation" : "Start"}
            </button>
          </div>
        </header>

        <section className="messages" aria-live="polite">
          {messages.length === 0 && (
            <p className="empty muted">
              {health ? "Press Start to talk to the advisor." : "Connecting to the advisor…"}
            </p>
          )}
          {messages.map((message, index) => (
            <div key={index} className={`bubble ${message.role}${message.unclear ? " unclear" : ""}`}>
              {message.text}
            </div>
          ))}
          {busy && <div className="bubble agent thinking">…</div>}
          <div ref={endRef} />
        </section>

        {error && (
          <div className="banner error" role="alert">
            {error}
          </div>
        )}
        {hint && <div className="banner hint">{hint}</div>}

        <footer className="controls">
          <button
            className={`talk${recorder.recording ? " recording" : ""}`}
            disabled={!canTalk && !recorder.recording}
            onPointerDown={onPointerDown}
            onPointerUp={() => void releaseTalk()}
            onPointerCancel={() => void releaseTalk()}
            onContextMenu={(e) => e.preventDefault()}
            title={voiceAvailable ? "Hold to talk (or hold Space)" : "Speech-to-text isn't configured on the server"}
          >
            {recorder.recording ? "Listening… release to send" : "Hold to talk"}
          </button>
          <form className="text-form" onSubmit={submitText}>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={turn ? "…or type your answer" : "Start a conversation first"}
              disabled={!turn || busy}
              maxLength={2000}
            />
            <button type="submit" disabled={!turn || busy || !draft.trim()}>
              Send
            </button>
          </form>
          {health && !voiceAvailable && (
            <p className="muted small">
              Voice is off: the server has no speech-to-text key. Typing still works.
            </p>
          )}
        </footer>
      </main>

      <DebugPanel turn={turn} health={health} />
    </div>
  );
}
