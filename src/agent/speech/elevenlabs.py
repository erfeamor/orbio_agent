"""ElevenLabs over the REST API: Flash v2.5 text-to-speech and Scribe speech-to-text."""

import math
from typing import Any

import httpx

from agent.speech.base import SpeechError, STTClient, Transcript, TTSClient

_BASE_URL = "https://api.elevenlabs.io/v1"


class ElevenLabsTTS(TTSClient):
    def __init__(
        self,
        api_key: str,
        *,
        voice_id: str,
        model: str = "eleven_flash_v2_5",
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def media_type(self) -> str:
        return "audio/mpeg"

    def synthesize(self, text: str) -> bytes:
        # Flash v2.5 detects the language from the text, so no language code is sent.
        try:
            response = self._client.post(
                f"{_BASE_URL}/text-to-speech/{self._voice_id}",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": self._api_key, "Accept": "audio/mpeg"},
                json={"text": text, "model_id": self._model},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SpeechError(f"ElevenLabs returned {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise SpeechError(f"cannot reach ElevenLabs: {exc}") from exc
        if not response.content:
            raise SpeechError("ElevenLabs returned no audio")
        return response.content


class ElevenLabsSTT(STTClient):
    """Scribe batch transcription; an alternative to Deepgram with the same API key as TTS."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "scribe_v2",
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def name(self) -> str:
        return "elevenlabs"

    def transcribe(self, audio: bytes, content_type: str) -> Transcript:
        try:
            response = self._client.post(
                f"{_BASE_URL}/speech-to-text",
                headers={"xi-api-key": self._api_key},
                data={"model_id": self._model},
                files={"file": ("clip", audio, content_type)},
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            text = str(body["text"]).strip()
        except httpx.HTTPStatusError as exc:
            raise SpeechError(f"ElevenLabs returned {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise SpeechError(f"cannot reach ElevenLabs: {exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise SpeechError("unexpected ElevenLabs response") from exc

        return Transcript(
            text=text,
            confidence=_confidence(body.get("words"), text),
            language=_iso_639_1(body.get("language_code")),
        )


def _confidence(words: object, text: str) -> float:
    """Scribe has no overall score: average the per-word probabilities (exp of logprob)."""
    if not text:
        return 0.0
    if not isinstance(words, list):
        return 1.0
    probabilities = [
        math.exp(float(word["logprob"]))
        for word in words
        if isinstance(word, dict) and word.get("type") == "word" and "logprob" in word
    ]
    if not probabilities:
        return 1.0
    return min(1.0, sum(probabilities) / len(probabilities))


def _iso_639_1(code: object) -> str | None:
    """Scribe may return 2- or 3-letter codes; only 2-letter ones match the engine's format."""
    if isinstance(code, str) and len(code.strip()) == 2:
        return code.strip().lower()
    return None
