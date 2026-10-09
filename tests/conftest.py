"""Shared fixtures. Every test runs isolated from the developer's .env and shell."""

import logging
from collections.abc import Iterator

import pytest

from agent.config import Settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings() in tests sees only defaults and what the test sets explicitly.

    Without this, a value in the developer's .env (or exported in their shell) could change
    or break unrelated tests.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture(autouse=True)
def restore_agent_logger() -> Iterator[None]:
    """configure_logging() changes global logger state; undo it after each test."""
    logger = logging.getLogger("agent")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    yield
    logger.handlers, logger.level, logger.propagate = saved[0], saved[1], saved[2]
