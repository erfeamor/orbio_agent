from collections.abc import Iterator
from pathlib import Path

import pytest
from factories import EN, make_engine, turn

from agent.config import Settings
from agent.domain.models import Intent
from agent.interfaces import cli
from agent.llm.base import LLMConfigurationError
from agent.llm.fake import FakeLLM
from agent.storage.file_repo import JsonFileRepository


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setenv("CONVERSATIONS_DIR", str(tmp_path))
    monkeypatch.setattr(
        cli, "build_engine", lambda s: make_engine(fake, JsonFileRepository(tmp_path))
    )
    return fake


def feed(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    inputs: Iterator[str] = iter(lines)

    def fake_input(prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


def test_chat_session_prints_replies_and_where_it_was_saved(
    llm: FakeLLM, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm.queue(turn(Intent.CONFIRM, reply="Great! Which year are you in?"))
    feed(monkeypatch, "yes", "exit")

    assert cli.main(["-v"]) == 0

    out = capsys.readouterr().out
    assert EN.consent in out
    assert "advisor> Great! Which year are you in?" in out
    assert "[phase=collecting" in out
    assert "Conversation saved to" in out


def test_end_of_input_exits_cleanly(
    llm: FakeLLM, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    feed(monkeypatch)
    assert cli.main([]) == 0


def test_configuration_error_exits_with_a_hint(
    llm: FakeLLM, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm.queue(LLMConfigurationError("no Anthropic credentials configured"))
    feed(monkeypatch, "yes")

    assert cli.main([]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_resuming_an_unknown_conversation_fails(
    llm: FakeLLM, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    feed(monkeypatch, "hello")

    assert cli.main(["--resume", "0" * 32]) == 1
    assert "no conversation" in capsys.readouterr().err


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("LLM_EFFORT", "medium")
    settings = Settings()
    assert settings.anthropic_model == "claude-sonnet-5-5"
    assert settings.llm_effort == "medium"
