"""Structured outputs the engine asks the LLM for. Each is sent as a JSON schema."""

from pydantic import BaseModel, Field

from agent.domain.models import Intent, ProfileField, Recommendation, Sentiment


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
    refused_fields: list[ProfileField] = Field(
        default_factory=list, description="Fields the student declined to answer this turn"
    )
    sentiment: Sentiment
    language: str = Field(description="ISO 639-1 code of the student's language, e.g. 'en'")


class ReplyOutput(BaseModel):
    """Structured output for calls that only produce a message to the student."""

    reply: str


class RecommendationOutput(BaseModel):
    recommendations: list[Recommendation]
    reply: str


class SummaryOutput(BaseModel):
    key_points: list[str]
