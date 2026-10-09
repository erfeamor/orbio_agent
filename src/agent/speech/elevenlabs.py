"""ElevenLabs Flash v2.5 text-to-speech over the REST API."""

import httpx

from agent.speech.base import SpeechError, TTSClient

_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


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
                _URL.format(voice_id=self._voice_id),
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
