"""Knowledge base of clubs and advanced classes, and the hard-requirement filter."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from agent.domain.models import (
    LearningModality,
    MagicOrientation,
    Specialization,
    StudentProfile,
)


class Category(StrEnum):
    CLUB = "CLUB"
    ADVANCED_CLASS = "ADVANCED_CLASS"


class IdealProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    magic_orientation: MagicOrientation
    learning_modality: LearningModality


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    category: Category
    description: str = Field(min_length=1)
    min_school_year: int = Field(ge=1, le=7)
    min_free_hours_required: float = Field(ge=0)
    excluded_specializations: list[Specialization] = Field(default_factory=list)
    ideal_profile: IdealProfile


class RejectionReason(StrEnum):
    SCHOOL_YEAR_TOO_LOW = "school_year_too_low"
    NOT_ENOUGH_FREE_HOURS = "not_enough_free_hours"
    SPECIALIZATION_EXCLUDED = "specialization_excluded"


@dataclass(frozen=True)
class Rejection:
    """Why an entry is closed to the student; ``required`` is the value the entry needs."""

    reason: RejectionReason
    required: int | float | str

    def __str__(self) -> str:
        # English wording for the LLM prompt; the LLM explains it in the student's language.
        match self.reason:
            case RejectionReason.SCHOOL_YEAR_TOO_LOW:
                return f"requires school year {self.required} or above"
            case RejectionReason.NOT_ENOUGH_FREE_HOURS:
                return f"requires {self.required:g} free hours per week"
            case RejectionReason.SPECIALIZATION_EXCLUDED:
                return f"not open to {self.required} students"


@dataclass
class FilterResult:
    eligible: list[CatalogEntry] = field(default_factory=list)
    rejected: dict[str, list[Rejection]] = field(default_factory=dict)


_catalog_adapter = TypeAdapter(list[CatalogEntry])


def load_catalog(path: Path) -> list[CatalogEntry]:
    entries = _catalog_adapter.validate_json(path.read_bytes())
    seen: set[str] = set()
    for entry in entries:
        key = entry.name.casefold()
        if key in seen:
            raise ValueError(f"duplicate catalog entry: {entry.name!r}")
        seen.add(key)
    return entries


def filter_catalog(entries: list[CatalogEntry], profile: StudentProfile) -> FilterResult:
    """Split the catalog into entries the student may join and the reasons for the rest."""
    if (
        profile.school_year is None
        or profile.specialization is None
        or profile.weekly_free_hours is None
    ):
        raise ValueError("school_year, specialization and weekly_free_hours are required")

    result = FilterResult()
    for entry in entries:
        reasons: list[Rejection] = []
        if profile.school_year < entry.min_school_year:
            reasons.append(Rejection(RejectionReason.SCHOOL_YEAR_TOO_LOW, entry.min_school_year))
        if profile.weekly_free_hours < entry.min_free_hours_required:
            reasons.append(
                Rejection(RejectionReason.NOT_ENOUGH_FREE_HOURS, entry.min_free_hours_required)
            )
        if profile.specialization in entry.excluded_specializations:
            reasons.append(
                Rejection(RejectionReason.SPECIALIZATION_EXCLUDED, profile.specialization.value)
            )

        if reasons:
            result.rejected[entry.name] = reasons
        else:
            result.eligible.append(entry)
    return result


@dataclass(frozen=True)
class RankedEntry:
    entry: CatalogEntry
    matches_orientation: bool
    matches_modality: bool

    @property
    def score(self) -> int:
        return int(self.matches_orientation) + int(self.matches_modality)


def rank_by_ideal_profile(
    entries: list[CatalogEntry], profile: StudentProfile, limit: int
) -> list[RankedEntry]:
    """Deterministic ranking by ideal-profile matches; the fallback when the LLM is unavailable.

    Returns what matched rather than sentences, so callers can word the reason in any language.
    """
    ranked = [
        RankedEntry(
            entry=entry,
            matches_orientation=entry.ideal_profile.magic_orientation is profile.magic_orientation,
            matches_modality=entry.ideal_profile.learning_modality is profile.learning_modality,
        )
        for entry in entries
    ]
    return sorted(ranked, key=lambda r: r.score, reverse=True)[:limit]
