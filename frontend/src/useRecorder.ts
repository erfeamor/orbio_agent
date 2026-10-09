import { useCallback, useEffect, useRef, useState } from "react";

// Shorter presses are treated as accidental taps.
const MIN_CLIP_MS = 400;

// Chrome/Firefox record WebM or Ogg Opus; Safari records MP4. Deepgram accepts all three.
const MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];

function pickMimeType(): string | undefined {
  return MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
}

/** Push-to-talk recording: start() on press, stop() on release resolves to the clip. */
export function useRecorder() {
  const [recording, setRecording] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef(0);
  // The first press waits for the microphone permission prompt; a release during
  // that wait must not leave a recorder running.
  const startingRef = useRef<Promise<void> | null>(null);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      throw new Error("This browser can't record audio. Microphone access needs https or localhost.");
    }
    const starting = (async () => {
      streamRef.current ??= await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(streamRef.current, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.start();
      recorderRef.current = recorder;
      startedAtRef.current = performance.now();
      setRecording(true);
    })();
    startingRef.current = starting;
    try {
      await starting;
    } finally {
      startingRef.current = null;
    }
  }, []);

  /** Stops recording. Resolves to null if the press was too short to be speech. */
  const stop = useCallback(async (): Promise<Blob | null> => {
    await startingRef.current?.catch(() => undefined);
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return null;
    return new Promise((resolve) => {
      recorder.onstop = () => {
        recorderRef.current = null;
        setRecording(false);
        const duration = performance.now() - startedAtRef.current;
        if (duration < MIN_CLIP_MS || chunksRef.current.length === 0) {
          resolve(null);
          return;
        }
        resolve(new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" }));
      };
      recorder.stop();
    });
  }, []);

  // Release the microphone when the page goes away.
  useEffect(() => () => streamRef.current?.getTracks().forEach((track) => track.stop()), []);

  return { recording, start, stop };
}
