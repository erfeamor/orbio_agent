"""Scripted speech clients for tests and offline runs."""

from collections.abc import Iterable

from agent.speech.base import SpeechError, STTClient, Transcript, TTSClient


class FakeSTT(STTClient):
    """Returns queued transcripts in order; a queued exception is raised instead."""

    def __init__(self, results: Iterable[Transcript | Exception] = ()) -> None:
        self._results = list(results)
        self.calls: list[tuple[bytes, str]] = []

    @property
    def name(self) -> str:
        return "fake"

    def queue(self, *results: Transcript | Exception) -> None:
        self._results.extend(results)

    def transcribe(self, audio: bytes, content_type: str) -> Transcript:
        self.calls.append((audio, content_type))
        if not self._results:
            raise AssertionError("FakeSTT has no transcript queued")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeTTS(TTSClient):
    """Returns b"audio:<text>", or raises SpeechError when ``fail`` is set."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    @property
    def media_type(self) -> str:
        return "audio/mpeg"

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        if self.fail:
            raise SpeechError("fake TTS failure")
        return f"audio:{text}".encode()
