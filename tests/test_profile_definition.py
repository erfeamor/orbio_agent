"""StudentProfile is the single definition of the profile; these keep the rest in sync."""

import pytest

from agent.domain.models import ConversationRecord, ProfileField, Role, StudentProfile
from agent.engine import prompts
from agent.engine.schemas import ExtractedFields


def test_profile_field_names_match_the_model_in_order() -> None:
    assert [f.value for f in ProfileField] == list(StudentProfile.model_fields)


def test_extracted_fields_mirror_the_profile() -> None:
    assert list(ExtractedFields.model_fields) == list(StudentProfile.model_fields)


def test_every_field_is_either_required_or_skippable() -> None:
    required = StudentProfile.required_fields()
    skippable = StudentProfile.skippable_fields()
    assert set(required) | set(skippable) == set(ProfileField)
    assert not set(required) & set(skippable)
    assert required == (
        ProfileField.SCHOOL_YEAR,
        ProfileField.SPECIALIZATION,
        ProfileField.WEEKLY_FREE_HOURS,
    )


def test_every_field_has_a_description_for_the_prompt() -> None:
    assert all(info.description for info in StudentProfile.model_fields.values())


@pytest.mark.parametrize(
    "line",
    [
        "- school_year: the student's year at the academy (integer from 1 to 7)",
        "- specialization: their branch of magic (one of ALCHEMY | CONJURING | DEFENSE | HEALING)",
        "- interests: what they enjoy or want to explore (free text)",
        "- weekly_free_hours: hours per week they could spend on clubs or classes "
        "(number from 0 to 168)",
    ],
)
def test_prompt_field_list_is_generated_from_the_model(line: str) -> None:
    assert line in prompts.INTERVIEW_SYSTEM_PROMPT


def test_state_travels_with_the_latest_student_message() -> None:
    record = ConversationRecord()
    record.add_message(Role.AGENT, "Shall we begin?")
    record.add_message(Role.STUDENT, "yes")

    messages = prompts.interview_messages(record, "Phase: consent")

    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert messages[1].content == "Shall we begin?"
    assert messages[2].content == (
        "<interview_state>\nPhase: consent\n</interview_state>\n\n"
        "<student_message>\nyes\n</student_message>"
    )


def test_interview_messages_require_a_student_message_last() -> None:
    record = ConversationRecord()
    record.add_message(Role.AGENT, "Shall we begin?")
    with pytest.raises(ValueError, match="student"):
        prompts.interview_messages(record, "state")


def test_system_prompt_does_not_change_with_the_conversation() -> None:
    """The cacheable prefix starts with the system prompt, so it must be a constant."""
    assert isinstance(prompts.INTERVIEW_SYSTEM_PROMPT, str)
    assert "Phase:" not in prompts.INTERVIEW_SYSTEM_PROMPT
