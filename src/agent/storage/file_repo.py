"""Conversation storage as one JSON file per conversation, for local use."""

import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from agent.domain.models import ConversationRecord
from agent.storage.base import (
    ConversationRepository,
    VersionConflictError,
    validate_conversation_id,
)


class JsonFileRepository(ConversationRepository):
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, conversation_id: str) -> Path:
        # Ids become file names, so only accept the format we generate (no path traversal).
        validate_conversation_id(conversation_id)
        return self._directory / f"{conversation_id}.json"

    def get(self, conversation_id: str) -> ConversationRecord | None:
        path = self._path(conversation_id)
        if not path.exists():
            return None
        return ConversationRecord.model_validate_json(path.read_bytes())

    def save(self, record: ConversationRecord) -> ConversationRecord:
        path = self._path(record.id)
        with self._lock:
            stored = self.get(record.id)
            stored_version = stored.version if stored else 0
            if stored_version != record.version:
                raise VersionConflictError.for_record(record, stored_version)
            saved = record.model_copy(
                update={"version": record.version + 1, "updated_at": datetime.now(UTC)}
            )
            # Write to a temp file and rename, so a crash never leaves a half-written record.
            fd, tmp = tempfile.mkstemp(dir=self._directory, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(saved.model_dump_json(indent=2))
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return saved
