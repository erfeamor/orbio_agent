"""Logging setup for the entry points (CLI, API server). Library code only gets loggers.

Every record carries the current ``conversation_id`` and ``request_id`` (from context
variables), so a turn can be followed across the engine, the LLM adapter and the speech
providers. ``json`` output is one JSON object per line, ready for CloudWatch Logs Insights.
"""

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Literal, TextIO

LogFormat = Literal["text", "json"]

conversation_id_var: ContextVar[str | None] = ContextVar("conversation_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_TEXT_FORMAT = "%(levelname)s %(name)s [conv=%(conversation_id)s req=%(request_id)s]: %(message)s"


@contextmanager
def bound(var: ContextVar[str | None], value: str) -> Iterator[None]:
    """Set a context variable for the duration of a block."""
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


class ContextFilter(logging.Filter):
    """Adds the current conversation and request ids to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.conversation_id = conversation_id_var.get() or "-"
        record.request_id = request_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "conversation_id": getattr(record, "conversation_id", "-"),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str, fmt: LogFormat, stream: TextIO | None = None) -> None:
    """Send this package's logs to ``stream`` (stderr by default). Safe to call twice."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(_TEXT_FORMAT))

    logger = logging.getLogger("agent")
    logger.handlers = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False
