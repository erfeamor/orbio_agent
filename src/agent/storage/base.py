"""Storage interface for conversation records."""

import re
from abc import ABC, abstractmethod

from agent.domain.models import ConversationRecord

# The format ConversationRecord generates (uuid4().hex). Ids become file names and S3 keys,
# so every repository, and the API path, accepts only this format.
CONVERSATION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def validate_conversation_id(conversation_id: str) -> str:
    if not CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
        raise ValueError(f"invalid conversation id: {conversation_id!r}")
    return conversation_id


class VersionConflictError(Exception):
    """The record was changed by someone else since it was loaded."""

    @classmethod
    def for_record(
        cls, record: ConversationRecord, stored_version: int | None
    ) -> "VersionConflictError":
        stored = (
            "no longer at the expected version"
            if stored_version is None
            else f"at version {stored_version}"
        )
        return cls(f"conversation {record.id} is {stored}, tried to save version {record.version}")


class StorageUnavailableError(Exception):
    """The storage backend failed (network, permissions, throttling); the turn can be retried."""


class ConversationRepository(ABC):
    """Loads and saves conversation records.

    ``save`` uses optimistic concurrency: it only succeeds if the stored version still
    matches ``record.version``, then increments the version. This keeps two concurrent
    turns of the same session from silently overwriting each other.
    """

    @abstractmethod
    def get(self, conversation_id: str) -> ConversationRecord | None: ...

    @abstractmethod
    def save(self, record: ConversationRecord) -> ConversationRecord: ...
