"""Domain models: the student profile, the LLM's per-turn output and the conversation record."""

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class Specialization(StrEnum):
    ALCHEMY = "ALCHEMY"
    CONJURING = "CONJURING"
    DEFENSE = "DEFENSE"
    HEALING = "HEALING"


class MagicOrientation(StrEnum):
    KNOWLEDGE = "KNOWLEDGE"
    POWER = "POWER"
    ALTRUISM = "ALTRUISM"
    ADVENTURE = "ADVENTURE"


class LearningModality(StrEnum):
    PRACTICAL = "PRACTICAL"
    THEORETICAL = "THEORETICAL"
    COLLABORATIVE = "COLLABORATIVE"


class Intent(StrEnum):
    ANSWER = "answer"
    CORRECTION = "correction"
    CONFIRM = "confirm"
    REJECT = "reject"
    OFF_TOPIC = "off_topic"
    REFUSE = "refuse"
    STOP = "stop"


class Sentiment(StrEnum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    FRUSTRATED = "frustrated"


class Phase(StrEnum):
    CONSENT = "consent"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    RECOMMENDING = "recommending"
    DONE = "done"


class StudentProfile(BaseModel):
    """What the interview collects. Every field is optional until the student provides it."""

    model_config = ConfigDict(extra="forbid")

    school_year: int | None = Field(default=None, ge=1, le=7)
    specialization: Specialization | None = None
    interests: str | None = Field(default=None, min_length=1, max_length=500)
    magic_orientation: MagicOrientation | None = None
    learning_modality: LearningModality | None = None
    weekly_free_hours: float | None = Field(default=None, ge=0, le=168)

    @field_validator("specialization", "magic_orientation", "learning_modality", mode="before")
    @classmethod
    def _normalize_enum(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("interests", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def missing_fields(self) -> list[str]:
        return [name for name in type(self).model_fields if getattr(self, name) is None]

    def is_complete(self) -> bool:
        return not self.missing_fields()

    def apply(self, updates: Mapping[str, Any]) -> tuple["StudentProfile", dict[str, str]]:
        """Validate each update on its own and merge the valid ones.

        Returns the new profile and an error message per rejected field, so one bad
        value never discards the good ones. ``None`` values mean "not mentioned".
        """
        accepted = self.model_dump()
        errors: dict[str, str] = {}
        for name, value in updates.items():
            if value is None:
                continue
            if name not in type(self).model_fields:
                errors[name] = "unknown field"
                continue
            try:
                type(self).model_validate({**accepted, name: value})
            except ValidationError as exc:
                errors[name] = exc.errors()[0]["msg"]
            else:
                accepted[name] = value
        return type(self).model_validate(accepted), errors


class ExtractedFields(BaseModel):
    """Values the LLM extracted from the latest message.

    Deliberately looser than StudentProfile: the LLM reports what the student said
    (e.g. specialization "necromancy") and StudentProfile.apply decides whether it is valid,
    so invalid answers can be reported back to the student instead of being coerced.
    """

    school_year: int | None = None
    specialization: str | None = None
    interests: str | None = None
    magic_orientation: str | None = None
    learning_modality: str | None = None
    weekly_free_hours: float | None = None


class TurnOutput(BaseModel):
    """Structured output the LLM returns for every interview turn."""

    extracted: ExtractedFields
    reply: str
    intent: Intent
    refused_fields: list[str] = Field(
        default_factory=list, description="Fields the student declined to answer this turn"
    )
    sentiment: Sentiment
    language: str = Field(description="ISO 639-1 code of the student's language, e.g. 'en'")


class Recommendation(BaseModel):
    name: str
    reason: str


class ReplyOutput(BaseModel):
    """Structured output for calls that only produce a message to the student."""

    reply: str


class RecommendationOutput(BaseModel):
    recommendations: list[Recommendation]
    reply: str


class SummaryOutput(BaseModel):
    key_points: list[str]


class Message(BaseModel):
    role: Literal["student", "agent"]
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationRecord(BaseModel):
    """One JSON document per session. It is both the stored conversation and the session state."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    version: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    phase: Phase = Phase.CONSENT
    language: str = "en"
    transcript: list[Message] = Field(default_factory=list)
    profile: StudentProfile = Field(default_factory=StudentProfile)
    field_errors: dict[str, str] = Field(default_factory=dict)
    skipped_fields: list[str] = Field(default_factory=list)
    sentiment_history: list[Sentiment] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    summary: list[str] | None = Field(default=None, description="Key points for school staff")
    model: str | None = None
    prompt_version: str | None = None

    def add_message(self, role: Literal["student", "agent"], text: str) -> None:
        self.transcript.append(Message(role=role, text=text))
