"""Deepgram Nova-3 speech-to-text over the REST API."""

from typing import Any

import httpx

from agent.speech.base import SpeechError, STTClient, Transcript

_URL = "https://api.deepgram.com/v1/listen"


class DeepgramSTT(STTClient):
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-3",
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def name(self) -> str:
        return "deepgram"

    def transcribe(self, audio: bytes, content_type: str) -> Transcript:
        try:
            response = self._client.post(
                _URL,
                params={"model": self._model, "smart_format": "true", "detect_language": "true"},
                headers={"Authorization": f"Token {self._api_key}", "Content-Type": content_type},
                content=audio,
            )
            response.raise_for_status()
            channel: dict[str, Any] = response.json()["results"]["channels"][0]
            best = channel["alternatives"][0]
        except httpx.HTTPStatusError as exc:
            raise SpeechError(f"Deepgram returned {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise SpeechError(f"cannot reach Deepgram: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise SpeechError("unexpected Deepgram response") from exc

        # Deepgram returns BCP-47 tags ("de-CH"); the engine works with ISO 639-1 ("de").
        detected = channel.get("detected_language")
        language = detected.split("-")[0].lower() if isinstance(detected, str) else None
        return Transcript(
            text=str(best.get("transcript", "")).strip(),
            confidence=float(best.get("confidence", 0.0)),
            language=language,
        )
