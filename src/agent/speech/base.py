"""Provider-independent speech-to-text and text-to-speech interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float
    language: str | None = None


class SpeechError(Exception):
    """A speech provider failed (network, timeout, bad response)."""


class STTClient(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name, reported by /health."""

    @abstractmethod
    def transcribe(self, audio: bytes, content_type: str) -> Transcript:
        """Transcribe one audio clip. Raises SpeechError on failure."""


class TTSClient(ABC):
    @property
    @abstractmethod
    def media_type(self) -> str: ...

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Return the spoken audio for ``text``. Raises SpeechError on failure."""
