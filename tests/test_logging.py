import io
import json
import logging
from pathlib import Path

import pytest
from factories import make_engine, turn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.domain.models import Intent
from agent.interfaces.api import REQUEST_ID_HEADER, Components, app_factory, create_app
from agent.llm.base import LLMUnavailableError
from agent.llm.fake import FakeLLM
from agent.logging_config import (
    bound,
    configure_logging,
    conversation_id_var,
    request_id_var,
)
from agent.storage.file_repo import JsonFileRepository

log = logging.getLogger("agent.test")


def capture(fmt: str = "json") -> io.StringIO:
    stream = io.StringIO()
    configure_logging("INFO", fmt, stream)  # type: ignore[arg-type]
    return stream


def test_json_lines_carry_the_context_ids() -> None:
    stream = capture()
    with bound(conversation_id_var, "conv-1"), bound(request_id_var, "req-1"):
        log.info("hello %s", "world")
    log.info("outside")

    inside, outside = (json.loads(line) for line in stream.getvalue().splitlines())
    assert inside["message"] == "hello world"
    assert (inside["conversation_id"], inside["request_id"]) == ("conv-1", "req-1")
    assert (outside["conversation_id"], outside["request_id"]) == ("-", "-")
    assert inside["level"] == "INFO" and inside["logger"] == "agent.test"


def test_json_includes_exceptions() -> None:
    stream = capture()
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed")
    entry = json.loads(stream.getvalue())
    assert "ValueError: boom" in entry["exception"]


def test_text_format_and_level() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", "text", stream)
    with bound(conversation_id_var, "conv-2"):
        log.info("hidden")
        log.warning("shown")
    assert stream.getvalue() == "WARNING agent.test [conv=conv-2 req=-]: shown\n"


def test_configuring_twice_does_not_duplicate_output() -> None:
    capture()
    stream = capture()
    log.info("once")
    assert len(stream.getvalue().splitlines()) == 1


def test_engine_logs_carry_the_conversation_id(tmp_path: Path) -> None:
    stream = capture()
    llm = FakeLLM()
    engine = make_engine(llm, JsonFileRepository(tmp_path))
    conversation_id = engine.start().record.id
    llm.queue(LLMUnavailableError("down"))

    engine.handle_turn(conversation_id, "hello")

    entries = [json.loads(line) for line in stream.getvalue().splitlines()]
    failure = next(e for e in entries if e["message"].startswith("turn failed"))
    assert failure["conversation_id"] == conversation_id


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    llm = FakeLLM([turn(Intent.CONFIRM)])
    app: FastAPI = create_app(
        Components(make_engine(llm, JsonFileRepository(tmp_path)), None, None)
    )
    return TestClient(app)


def test_api_returns_a_request_id(client: TestClient) -> None:
    response = client.get("/health")
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_api_keeps_a_valid_incoming_request_id(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "abc-123"})
    assert response.headers[REQUEST_ID_HEADER] == "abc-123"


def test_api_replaces_an_unsafe_incoming_request_id(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "evil\nINFO fake log line"})
    assert response.headers[REQUEST_ID_HEADER] != "evil\nINFO fake log line"
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_app_factory_configures_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CONVERSATIONS_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_FORMAT", "json")

    app = app_factory()

    assert isinstance(app, FastAPI)
    handler = logging.getLogger("agent").handlers[0]
    assert type(handler.formatter).__name__ == "JsonFormatter"
