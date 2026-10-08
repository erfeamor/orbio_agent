import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.domain.catalog import CatalogEntry, filter_catalog, load_catalog
from agent.domain.models import Specialization, StudentProfile

CATALOG_PATH = Path(__file__).parents[1] / "data" / "catalog.json"


def make_entry(name: str = "Test Club", **overrides: object) -> CatalogEntry:
    data: dict[str, object] = {
        "name": name,
        "category": "CLUB",
        "description": "A club.",
        "min_school_year": 1,
        "min_free_hours_required": 0,
        "excluded_specializations": [],
        "ideal_profile": {"magic_orientation": "KNOWLEDGE", "learning_modality": "PRACTICAL"},
    }
    data.update(overrides)
    return CatalogEntry.model_validate(data)


def make_profile(**overrides: object) -> StudentProfile:
    data: dict[str, object] = {
        "school_year": 4,
        "specialization": "DEFENSE",
        "weekly_free_hours": 5,
    }
    data.update(overrides)
    return StudentProfile.model_validate(data)


def test_bundled_catalog_is_valid() -> None:
    entries = load_catalog(CATALOG_PATH)
    assert len(entries) == 15


def test_duplicate_names_are_rejected(tmp_path: Path) -> None:
    entry = make_entry().model_dump(mode="json")
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps([entry, {**entry, "name": "test club"}]))
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(path)


def test_invalid_entry_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps([{**make_entry().model_dump(mode="json"), "min_school_year": 9}]))
    with pytest.raises(ValidationError):
        load_catalog(path)


def test_filter_keeps_entries_that_meet_all_requirements() -> None:
    entry = make_entry(min_school_year=4, min_free_hours_required=5)
    result = filter_catalog([entry], make_profile())
    assert result.eligible == [entry]
    assert result.rejected == {}


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"min_school_year": 5}, "requires school year 5 or above"),
        ({"min_free_hours_required": 5.5}, "requires 5.5 free hours per week"),
        ({"excluded_specializations": ["DEFENSE"]}, "not open to DEFENSE students"),
    ],
)
def test_filter_rejects_with_reason(overrides: dict[str, object], reason: str) -> None:
    result = filter_catalog([make_entry(**overrides)], make_profile())
    assert result.eligible == []
    assert result.rejected == {"Test Club": [reason]}


def test_filter_collects_every_failed_requirement() -> None:
    entry = make_entry(
        min_school_year=7, min_free_hours_required=10, excluded_specializations=["DEFENSE"]
    )
    result = filter_catalog([entry], make_profile())
    assert len(result.rejected["Test Club"]) == 3


def test_filter_can_reject_everything() -> None:
    profile = make_profile(
        school_year=1, weekly_free_hours=0, specialization=Specialization.HEALING
    )
    result = filter_catalog(load_catalog(CATALOG_PATH), profile)
    assert result.eligible == []
    assert len(result.rejected) == 15


def test_filter_requires_the_fields_it_checks() -> None:
    with pytest.raises(ValueError, match="required"):
        filter_catalog([make_entry()], StudentProfile(school_year=3))
