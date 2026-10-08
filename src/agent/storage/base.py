"""Storage interface for conversation records."""

from abc import ABC, abstractmethod

from agent.domain.models import ConversationRecord


class VersionConflictError(Exception):
    """The record was changed by someone else since it was loaded."""


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
