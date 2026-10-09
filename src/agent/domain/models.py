"""Domain models: the student profile and the conversation record."""

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_LANGUAGE = "en"


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
    DONE = "done"


class EndReason(StrEnum):
    DECLINED = "declined"  # said no to the consent question: no interview took place
    STOPPED = "stopped"  # asked to end before the recommendations
    COMPLETED = "completed"  # confirmed the profile and got recommendations


class Role(StrEnum):
    STUDENT = "student"
    AGENT = "agent"


class ProfileField(StrEnum):
    """Names of the StudentProfile fields, in the order the interview asks for them."""

    SCHOOL_YEAR = "school_year"
    SPECIALIZATION = "specialization"
    INTERESTS = "interests"
    MAGIC_ORIENTATION = "magic_orientation"
    LEARNING_MODALITY = "learning_modality"
    WEEKLY_FREE_HOURS = "weekly_free_hours"


def _skippable(skippable: bool) -> dict[str, Any]:
    """Field metadata: may the student decline to answer? Required fields feed hard filters."""
    return {"skippable": skippable}


class StudentProfile(BaseModel):
    """What the interview collects. Every field is optional until the student provides it.

    This is the single definition of the profile: the prompt's field list (descriptions and
    limits) and the required/skippable split are derived from it.
    """

    model_config = ConfigDict(extra="forbid")

    school_year: int | None = Field(
        default=None,
        ge=1,
        le=7,
        description="the student's year at the academy",
        json_schema_extra=_skippable(False),
    )
    specialization: Specialization | None = Field(
        default=None,
        description="their branch of magic",
        json_schema_extra=_skippable(False),
    )
    interests: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="what they enjoy or want to explore",
        json_schema_extra=_skippable(True),
    )
    magic_orientation: MagicOrientation | None = Field(
        default=None,
        description="what drives their magic",
        json_schema_extra=_skippable(True),
    )
    learning_modality: LearningModality | None = Field(
        default=None,
        description="how they learn best",
        json_schema_extra=_skippable(True),
    )
    weekly_free_hours: float | None = Field(
        default=None,
        ge=0,
        le=168,
        description="hours per week they could spend on clubs or classes",
        json_schema_extra=_skippable(False),
    )

    @classmethod
    def skippable_fields(cls) -> tuple[ProfileField, ...]:
        return tuple(ProfileField(name) for name in cls.model_fields if cls._is_skippable(name))

    @classmethod
    def required_fields(cls) -> tuple[ProfileField, ...]:
        return tuple(ProfileField(name) for name in cls.model_fields if not cls._is_skippable(name))

    @classmethod
    def _is_skippable(cls, name: str) -> bool:
        extra = cls.model_fields[name].json_schema_extra
        return isinstance(extra, dict) and extra.get("skippable") is True

    @field_validator("specialization", "magic_orientation", "learning_modality", mode="before")
    @classmethod
    def _normalize_enum(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("interests", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def missing_fields(self) -> list[ProfileField]:
        return [
            ProfileField(name) for name in type(self).model_fields if getattr(self, name) is None
        ]

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


class Recommendation(BaseModel):
    name: str
    reason: str


class Message(BaseModel):
    role: Role
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationRecord(BaseModel):
    """One JSON document per session. It is both the stored conversation and the session state."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    version: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    phase: Phase = Phase.CONSENT
    end_reason: EndReason | None = Field(default=None, description="Set when phase is done")
    language: str = DEFAULT_LANGUAGE
    transcript: list[Message] = Field(default_factory=list)
    profile: StudentProfile = Field(default_factory=StudentProfile)
    field_errors: dict[str, str] = Field(default_factory=dict)
    skipped_fields: list[ProfileField] = Field(default_factory=list)
    sentiment_history: list[Sentiment] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    summary: list[str] | None = Field(default=None, description="Key points for school staff")
    model: str | None = None
    prompt_version: str | None = None

    def add_message(self, role: Role, text: str) -> None:
        self.transcript.append(Message(role=role, text=text))
