import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.config import Settings
from agent.domain.catalog import load_catalog
from agent.domain.models import ExtractedFields, Intent, Sentiment, TurnOutput
from agent.engine import prompts
from agent.engine.engine import Engine
from agent.interfaces.api import Components, create_app
from agent.llm.base import LLMConfigurationError
from agent.llm.fake import FakeLLM
from agent.speech.base import Transcript
from agent.speech.fake import FakeSTT, FakeTTS
from agent.storage.base import VersionConflictError
from agent.storage.file_repo import JsonFileRepository

CATALOG = load_catalog(Path(__file__).parents[1] / "data" / "catalog.json")
UNKNOWN_ID = "0" * 32


def turn(intent: Intent = Intent.ANSWER, **extracted: object) -> TurnOutput:
    return TurnOutput(
        extracted=ExtractedFields(**extracted),  # type: ignore[arg-type]
        reply=f"reply ({intent.value})",
        intent=intent,
        sentiment=Sentiment.NEUTRAL,
        language="en",
    )


class Harness:
    def __init__(self, tmp_path: Path, *, stt: bool = True, tts: bool = True) -> None:
        self.llm = FakeLLM()
        self.stt = FakeSTT() if stt else None
        self.tts = FakeTTS() if tts else None
        self.engine = Engine(self.llm, JsonFileRepository(tmp_path), CATALOG)
        settings = Settings(_env_file=None, max_audio_bytes=100)  # type: ignore[call-arg]
        app = create_app(Components(self.engine, self.stt, self.tts), settings)
        self.client = TestClient(app)

    def start(self) -> str:
        response = self.client.post("/sessions")
        assert response.status_code == 201
        return str(response.json()["conversation_id"])


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def upload(
    data: bytes = b"clip", content_type: str = "audio/webm;codecs=opus"
) -> dict[str, object]:
    return {"audio": ("clip.webm", data, content_type)}


def test_health_reports_speech_availability(tmp_path: Path) -> None:
    body = Harness(tmp_path, stt=False).client.get("/health").json()
    assert body == {
        "status": "ok",
        "speech_to_text": False,
        "stt_provider": None,
        "text_to_speech": True,
    }


def test_health_names_the_stt_provider(h: Harness) -> None:
    assert h.client.get("/health").json()["stt_provider"] == "fake"


def test_start_returns_the_consent_message(h: Harness) -> None:
    response = h.client.post("/sessions", json={"speak": True})

    body = response.json()
    assert response.status_code == 201
    assert body["reply"] == prompts.CONSENT_MESSAGE
    assert body["phase"] == "consent"
    assert base64.b64decode(body["audio_base64"]) == f"audio:{prompts.CONSENT_MESSAGE}".encode()
    assert body["audio_media_type"] == "audio/mpeg"


def test_text_message_runs_a_turn(h: Harness) -> None:
    conversation_id = h.start()
    h.llm.queue(turn(Intent.CONFIRM))
    h.client.post(f"/sessions/{conversation_id}/messages", json={"text": "yes"})
    h.llm.queue(turn(school_year=3))

    response = h.client.post(f"/sessions/{conversation_id}/messages", json={"text": "year 3"})

    body = response.json()
    assert response.status_code == 200
    assert body["reply"] == "reply (answer)"
    assert body["phase"] == "collecting"
    assert body["profile"]["school_year"] == 3
    assert body["audio_base64"] is None


def test_audio_message_returns_transcript_reply_and_audio(h: Harness) -> None:
    conversation_id = h.start()
    assert h.stt is not None
    h.stt.queue(Transcript("yes please", 0.9, "en"))
    h.llm.queue(turn(Intent.CONFIRM))

    response = h.client.post(f"/sessions/{conversation_id}/audio", files=upload())

    body = response.json()
    assert response.status_code == 200
    assert body["transcript"] == "yes please"
    assert body["reply"] == "reply (confirm)"
    assert base64.b64decode(body["audio_base64"]) == b"audio:reply (confirm)"
    assert h.stt.calls == [(b"clip", "audio/webm")]


def test_get_session_returns_the_full_record(h: Harness) -> None:
    conversation_id = h.start()
    body = h.client.get(f"/sessions/{conversation_id}").json()
    assert body["id"] == conversation_id
    assert body["transcript"][0]["text"] == prompts.CONSENT_MESSAGE


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("get", f"/sessions/{UNKNOWN_ID}", {}),
        ("post", f"/sessions/{UNKNOWN_ID}/messages", {"json": {"text": "hi"}}),
        ("post", f"/sessions/{UNKNOWN_ID}/audio", {"files": upload()}),
    ],
)
def test_unknown_conversation_is_404(
    h: Harness, method: str, path: str, kwargs: dict[str, object]
) -> None:
    assert getattr(h.client, method)(path, **kwargs).status_code == 404


def test_malformed_conversation_id_is_rejected(h: Harness) -> None:
    response = h.client.post("/sessions/..%2Fetc/messages", json={"text": "hi"})
    assert response.status_code in (404, 422)


@pytest.mark.parametrize(
    ("data", "content_type", "status"),
    [
        (b"clip", "text/plain", 415),
        (b"x" * 101, "audio/webm", 413),
        (b"", "audio/webm", 400),
    ],
)
def test_bad_audio_uploads_are_rejected(
    h: Harness, data: bytes, content_type: str, status: int
) -> None:
    conversation_id = h.start()
    response = h.client.post(f"/sessions/{conversation_id}/audio", files=upload(data, content_type))
    assert response.status_code == status


def test_audio_without_stt_configured_is_503(tmp_path: Path) -> None:
    harness = Harness(tmp_path, stt=False)
    conversation_id = harness.start()
    response = harness.client.post(f"/sessions/{conversation_id}/audio", files=upload())
    assert response.status_code == 503


def test_llm_configuration_error_is_503_without_details(h: Harness) -> None:
    conversation_id = h.start()
    h.llm.queue(LLMConfigurationError("secret detail: bad key sk-..."))

    response = h.client.post(f"/sessions/{conversation_id}/messages", json={"text": "hi"})

    assert response.status_code == 503
    assert "sk-" not in response.text


def test_version_conflict_is_409(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    conversation_id = h.start()

    def conflict(conversation_id: str, text: str) -> None:
        raise VersionConflictError("stale")

    monkeypatch.setattr(h.engine, "handle_turn", conflict)
    response = h.client.post(f"/sessions/{conversation_id}/messages", json={"text": "hi"})
    assert response.status_code == 409


def test_cors_allows_the_frontend_origin(h: Harness) -> None:
    response = h.client.options(
        "/sessions",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_other_origins(h: Harness) -> None:
    response = h.client.options(
        "/sessions",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers
