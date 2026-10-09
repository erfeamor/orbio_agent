"""Knowledge base of clubs and advanced classes, and the hard-requirement filter."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from agent.domain.models import (
    LearningModality,
    MagicOrientation,
    Recommendation,
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


@dataclass
class FilterResult:
    eligible: list[CatalogEntry] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)


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
        reasons: list[str] = []
        if profile.school_year < entry.min_school_year:
            reasons.append(f"requires school year {entry.min_school_year} or above")
        if profile.weekly_free_hours < entry.min_free_hours_required:
            reasons.append(f"requires {entry.min_free_hours_required:g} free hours per week")
        if profile.specialization in entry.excluded_specializations:
            reasons.append(f"not open to {profile.specialization.value} students")

        if reasons:
            result.rejected[entry.name] = reasons
        else:
            result.eligible.append(entry)
    return result


def rank_by_ideal_profile(
    entries: list[CatalogEntry], profile: StudentProfile, limit: int
) -> list[Recommendation]:
    """Deterministic ranking by ideal-profile matches; the fallback when the LLM is unavailable."""

    def matches(entry: CatalogEntry) -> list[str]:
        found = []
        if entry.ideal_profile.magic_orientation is profile.magic_orientation:
            found.append(f"suits your {entry.ideal_profile.magic_orientation.value.lower()} drive")
        if entry.ideal_profile.learning_modality is profile.learning_modality:
            found.append(f"fits your {entry.ideal_profile.learning_modality.value.lower()} style")
        return found

    ranked = sorted(entries, key=lambda e: len(matches(e)), reverse=True)[:limit]
    return [
        Recommendation(
            name=entry.name,
            reason=" and ".join(matches(entry)).capitalize() or "Open to you",
        )
        for entry in ranked
    ]
