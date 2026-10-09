import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent.domain.catalog import load_catalog
from agent.domain.models import ExtractedFields, Intent, Sentiment, TurnOutput
from agent.engine import prompts
from agent.engine.engine import ConversationNotFoundError, Engine
from agent.llm.fake import FakeLLM
from agent.speech.base import SpeechError, Transcript
from agent.speech.deepgram import DeepgramSTT
from agent.speech.elevenlabs import ElevenLabsSTT, ElevenLabsTTS
from agent.speech.fake import FakeSTT, FakeTTS
from agent.speech.pipeline import HEARING_FAILURE_MESSAGE, VoicePipeline
from agent.storage.file_repo import JsonFileRepository

CATALOG = load_catalog(Path(__file__).parents[1] / "data" / "catalog.json")


def mock_client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def deepgram_body(transcript: str, confidence: float, language: str | None) -> dict[str, Any]:
    channel: dict[str, Any] = {
        "alternatives": [{"transcript": transcript, "confidence": confidence}]
    }
    if language:
        channel["detected_language"] = language
    return {"results": {"channels": [channel]}}


# --- Deepgram ---


def test_deepgram_sends_audio_and_parses_the_transcript() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=deepgram_body(" año tres ", 0.93, "es-ES"))

    stt = DeepgramSTT("dg-key", client=mock_client(handler))
    result = stt.transcribe(b"webm-bytes", "audio/webm")

    assert result == Transcript(text="año tres", confidence=0.93, language="es")
    assert stt.name == "deepgram"
    request = seen[0]
    assert request.headers["Authorization"] == "Token dg-key"
    assert request.headers["Content-Type"] == "audio/webm"
    assert request.url.params["model"] == "nova-3"
    assert request.url.params["detect_language"] == "true"
    assert request.content == b"webm-bytes"


def test_deepgram_without_detected_language() -> None:
    stt = DeepgramSTT(
        "k", client=mock_client(lambda r: httpx.Response(200, json=deepgram_body("hi", 0.9, None)))
    )
    assert stt.transcribe(b"x", "audio/wav").language is None


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401, json={}), "Deepgram returned 401"),
        (httpx.Response(200, json={"results": {"channels": []}}), "unexpected Deepgram response"),
        (httpx.Response(200, content=b"not json"), "unexpected Deepgram response"),
    ],
)
def test_deepgram_errors_become_speech_errors(response: httpx.Response, message: str) -> None:
    stt = DeepgramSTT("k", client=mock_client(lambda r: response))
    with pytest.raises(SpeechError, match=message):
        stt.transcribe(b"x", "audio/wav")


def test_deepgram_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    with pytest.raises(SpeechError, match="cannot reach Deepgram"):
        DeepgramSTT("k", client=mock_client(handler)).transcribe(b"x", "audio/wav")


# --- ElevenLabs ---


def test_elevenlabs_returns_mp3_bytes() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"ID3mp3")

    tts = ElevenLabsTTS("el-key", voice_id="voice123", client=mock_client(handler))

    assert tts.synthesize("Hola") == b"ID3mp3"
    assert tts.media_type == "audio/mpeg"
    request = seen[0]
    assert request.url.path == "/v1/text-to-speech/voice123"
    assert request.headers["xi-api-key"] == "el-key"
    assert json.loads(request.content) == {"text": "Hola", "model_id": "eleven_flash_v2_5"}


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(429, json={}), "ElevenLabs returned 429"),
        (httpx.Response(200, content=b""), "no audio"),
    ],
)
def test_elevenlabs_errors_become_speech_errors(response: httpx.Response, message: str) -> None:
    tts = ElevenLabsTTS("k", voice_id="v", client=mock_client(lambda r: response))
    with pytest.raises(SpeechError, match=message):
        tts.synthesize("hi")


def test_elevenlabs_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    tts = ElevenLabsTTS("k", voice_id="v", client=mock_client(handler))
    with pytest.raises(SpeechError, match="cannot reach ElevenLabs"):
        tts.synthesize("hi")


# --- ElevenLabs Scribe (STT) ---


def scribe_body(text: str, logprobs: list[float], language: str | None = "es") -> dict[str, Any]:
    words: list[dict[str, Any]] = []
    for index, logprob in enumerate(logprobs):
        words.append({"text": f"w{index}", "type": "word", "logprob": logprob})
        words.append({"text": " ", "type": "spacing", "logprob": -9.0})
    return {"text": text, "language_code": language, "language_probability": 0.99, "words": words}


def test_scribe_uploads_the_clip_and_parses_the_transcript() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=scribe_body(" año tres ", [0.0, 0.0]))

    stt = ElevenLabsSTT("el-key", client=mock_client(handler))
    result = stt.transcribe(b"webm-bytes", "audio/webm")

    assert result == Transcript(text="año tres", confidence=1.0, language="es")
    assert stt.name == "elevenlabs"
    request = seen[0]
    assert request.url.path == "/v1/speech-to-text"
    assert request.headers["xi-api-key"] == "el-key"
    assert request.headers["Content-Type"].startswith("multipart/form-data")
    body = request.content
    assert b'name="model_id"' in body and b"scribe_v2" in body
    assert b'name="file"' in body and b"webm-bytes" in body
    assert b"Content-Type: audio/webm" in body


def test_scribe_confidence_averages_word_probabilities_ignoring_spacing() -> None:
    body = scribe_body("two words", [math.log(0.9), math.log(0.5)])
    stt = ElevenLabsSTT("k", client=mock_client(lambda r: httpx.Response(200, json=body)))
    assert stt.transcribe(b"x", "audio/webm").confidence == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("body", "confidence", "language"),
    [
        ({"text": "", "words": []}, 0.0, None),
        ({"text": "hello"}, 1.0, None),
        ({"text": "hello", "words": [{"type": "spacing"}]}, 1.0, None),
        ({"text": "hello", "language_code": "eng", "words": []}, 1.0, None),
        ({"text": "hello", "language_code": "EN", "words": []}, 1.0, "en"),
    ],
    ids=["empty", "no-words", "no-scored-words", "three-letter-code", "two-letter-code"],
)
def test_scribe_edge_cases(body: dict[str, Any], confidence: float, language: str | None) -> None:
    stt = ElevenLabsSTT("k", client=mock_client(lambda r: httpx.Response(200, json=body)))
    result = stt.transcribe(b"x", "audio/webm")
    assert (result.confidence, result.language) == (confidence, language)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401, json={}), "ElevenLabs returned 401"),
        (httpx.Response(200, json={"words": []}), "unexpected ElevenLabs response"),
        (httpx.Response(200, content=b"not json"), "unexpected ElevenLabs response"),
    ],
)
def test_scribe_errors_become_speech_errors(response: httpx.Response, message: str) -> None:
    stt = ElevenLabsSTT("k", client=mock_client(lambda r: response))
    with pytest.raises(SpeechError, match=message):
        stt.transcribe(b"x", "audio/webm")


def test_scribe_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(SpeechError, match="cannot reach ElevenLabs"):
        ElevenLabsSTT("k", client=mock_client(handler)).transcribe(b"x", "audio/webm")


# --- Voice pipeline ---


@pytest.fixture
def engine_and_llm(tmp_path: Path) -> tuple[Engine, FakeLLM]:
    llm = FakeLLM()
    return Engine(llm, JsonFileRepository(tmp_path), CATALOG), llm


def confirm_turn() -> TurnOutput:
    return TurnOutput(
        extracted=ExtractedFields(),
        reply="Great! Which year are you in?",
        intent=Intent.CONFIRM,
        sentiment=Sentiment.NEUTRAL,
        language="en",
    )


def test_pipeline_runs_stt_engine_and_tts(engine_and_llm: tuple[Engine, FakeLLM]) -> None:
    engine, llm = engine_and_llm
    conversation_id = engine.start().record.id
    stt, tts = FakeSTT([Transcript("yes please", 0.95, "en")]), FakeTTS()
    llm.queue(confirm_turn())

    turn = VoicePipeline(engine, stt, tts).handle(conversation_id, b"clip", "audio/webm")

    assert turn.transcript == "yes please"
    assert turn.reply == "Great! Which year are you in?"
    assert turn.audio == b"audio:Great! Which year are you in?"
    assert turn.record.transcript[-2].text == "yes please"


@pytest.mark.parametrize(
    "transcript", [Transcript("", 0.0), Transcript("mumble", 0.2)], ids=["empty", "low-confidence"]
)
def test_unclear_audio_asks_again_without_a_turn(
    engine_and_llm: tuple[Engine, FakeLLM], transcript: Transcript
) -> None:
    engine, llm = engine_and_llm
    start = engine.start().record

    turn = VoicePipeline(engine, FakeSTT([transcript]), FakeTTS()).handle(
        start.id, b"clip", "audio/webm"
    )

    assert turn.reply == prompts.EMPTY_INPUT_MESSAGE
    assert turn.record.version == start.version
    assert llm.calls == []


def test_stt_failure_asks_again(engine_and_llm: tuple[Engine, FakeLLM]) -> None:
    engine, _ = engine_and_llm
    conversation_id = engine.start().record.id
    stt = FakeSTT([SpeechError("down")])

    turn = VoicePipeline(engine, stt, FakeTTS()).handle(conversation_id, b"clip", "audio/webm")

    assert turn.reply == HEARING_FAILURE_MESSAGE
    assert turn.transcript is None


def test_tts_failure_still_returns_the_text(engine_and_llm: tuple[Engine, FakeLLM]) -> None:
    engine, llm = engine_and_llm
    conversation_id = engine.start().record.id
    llm.queue(confirm_turn())
    stt = FakeSTT([Transcript("yes", 0.9)])

    turn = VoicePipeline(engine, stt, FakeTTS(fail=True)).handle(
        conversation_id, b"clip", "audio/webm"
    )

    assert turn.reply == "Great! Which year are you in?"
    assert turn.audio is None


def test_unknown_conversation_is_rejected_before_transcribing(
    engine_and_llm: tuple[Engine, FakeLLM],
) -> None:
    engine, _ = engine_and_llm
    stt = FakeSTT()
    with pytest.raises(ConversationNotFoundError):
        VoicePipeline(engine, stt, None).handle("0" * 32, b"clip", "audio/webm")
    assert stt.calls == []
