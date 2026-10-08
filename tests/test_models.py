import pytest
from pydantic import ValidationError

from agent.domain.models import (
    ConversationRecord,
    LearningModality,
    Phase,
    Specialization,
    StudentProfile,
)


def test_new_profile_is_empty_and_lists_all_fields_as_missing() -> None:
    profile = StudentProfile()
    assert profile.missing_fields() == [
        "school_year",
        "specialization",
        "interests",
        "magic_orientation",
        "learning_modality",
        "weekly_free_hours",
    ]
    assert not profile.is_complete()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("school_year", 0),
        ("school_year", 8),
        ("school_year", 3.5),
        ("specialization", "NECROMANCY"),
        ("magic_orientation", "CHAOS"),
        ("learning_modality", "osmosis"),
        ("weekly_free_hours", -1),
        ("weekly_free_hours", 200),
        ("interests", "   "),
    ],
)
def test_invalid_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        StudentProfile.model_validate({field: value})


def test_enum_values_are_case_and_whitespace_insensitive() -> None:
    profile = StudentProfile(specialization=" alchemy ", learning_modality="Practical")  # type: ignore[arg-type]
    assert profile.specialization is Specialization.ALCHEMY
    assert profile.learning_modality is LearningModality.PRACTICAL


def test_apply_merges_valid_updates_and_reports_invalid_ones() -> None:
    profile, errors = StudentProfile().apply(
        {"school_year": 9, "specialization": "healing", "weekly_free_hours": 6}
    )
    assert profile.school_year is None
    assert profile.specialization is Specialization.HEALING
    assert profile.weekly_free_hours == 6
    assert set(errors) == {"school_year"}
    assert "less than or equal to 7" in errors["school_year"]


def test_apply_ignores_none_and_keeps_existing_values() -> None:
    profile = StudentProfile(school_year=3)
    updated, errors = profile.apply({"school_year": None, "interests": "potions"})
    assert updated.school_year == 3
    assert updated.interests == "potions"
    assert errors == {}


def test_apply_overwrites_a_value_when_the_student_corrects_it() -> None:
    profile = StudentProfile(school_year=3)
    updated, _ = profile.apply({"school_year": 4})
    assert updated.school_year == 4


def test_apply_reports_unknown_fields() -> None:
    _, errors = StudentProfile().apply({"wand_type": "oak"})
    assert errors == {"wand_type": "unknown field"}


def test_complete_profile() -> None:
    profile = StudentProfile(
        school_year=5,
        specialization=Specialization.DEFENSE,
        interests="duels and wards",
        magic_orientation="POWER",  # type: ignore[arg-type]
        learning_modality="PRACTICAL",  # type: ignore[arg-type]
        weekly_free_hours=5,
    )
    assert profile.is_complete()


def test_conversation_record_round_trips_through_json() -> None:
    record = ConversationRecord()
    record.add_message("agent", "Welcome!")
    record.add_message("student", "Hi, I'm in year 3")
    record.profile, record.field_errors = record.profile.apply({"school_year": 3})

    restored = ConversationRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.phase is Phase.CONSENT
    assert [m.role for m in restored.transcript] == ["agent", "student"]
