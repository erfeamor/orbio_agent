from pathlib import Path
from typing import Any

import pytest

from agent.domain.catalog import load_catalog
from agent.domain.models import (
    ExtractedFields,
    Intent,
    Phase,
    Recommendation,
    RecommendationOutput,
    ReplyOutput,
    Sentiment,
    SummaryOutput,
    TurnOutput,
)
from agent.engine import prompts
from agent.engine.engine import ConversationNotFoundError, Engine
from agent.llm.base import LLMOutputError, LLMRefusalError, LLMUnavailableError
from agent.llm.fake import FakeLLM
from agent.storage.base import VersionConflictError
from agent.storage.file_repo import JsonFileRepository

CATALOG = load_catalog(Path(__file__).parents[1] / "data" / "catalog.json")

FULL_PROFILE: dict[str, Any] = {
    "school_year": 4,
    "specialization": "DEFENSE",
    "interests": "dueling",
    "magic_orientation": "POWER",
    "learning_modality": "PRACTICAL",
    "weekly_free_hours": 5,
}


def turn(
    intent: Intent = Intent.ANSWER,
    reply: str = "llm reply",
    sentiment: Sentiment = Sentiment.NEUTRAL,
    language: str = "en",
    refused: list[str] | None = None,
    **extracted: Any,
) -> TurnOutput:
    return TurnOutput(
        extracted=ExtractedFields(**extracted),
        reply=reply,
        intent=intent,
        sentiment=sentiment,
        language=language,
        refused_fields=refused or [],
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def engine(llm: FakeLLM, tmp_path: Path) -> Engine:
    return Engine(llm, JsonFileRepository(tmp_path), CATALOG)


def start_collecting(engine: Engine, llm: FakeLLM) -> str:
    conversation_id = engine.start().record.id
    llm.queue(turn(Intent.CONFIRM, reply="Great! Which year are you in?"))
    engine.handle_turn(conversation_id, "yes")
    return conversation_id


def reach_confirmation(engine: Engine, llm: FakeLLM) -> str:
    conversation_id = start_collecting(engine, llm)
    llm.queue(turn(**FULL_PROFILE), ReplyOutput(reply="Is this right?"))
    result = engine.handle_turn(conversation_id, "everything at once")
    assert result.record.phase is Phase.CONFIRMING
    return conversation_id


def test_start_saves_a_conversation_with_the_consent_message(engine: Engine) -> None:
    result = engine.start()
    assert result.reply == prompts.CONSENT_MESSAGE
    assert result.record.phase is Phase.CONSENT
    assert result.record.version == 1
    assert result.record.model == "fake"


def test_full_interview_from_consent_to_recommendations(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)

    llm.queue(turn(school_year=4, specialization="defense"))
    result = engine.handle_turn(conversation_id, "4th year, defense")
    assert result.record.phase is Phase.COLLECTING
    assert result.record.profile.school_year == 4

    llm.queue(turn(interests="dueling", magic_orientation="POWER"))
    engine.handle_turn(conversation_id, "I love dueling, I want to be powerful")

    llm.queue(
        turn(learning_modality="PRACTICAL", weekly_free_hours=5),
        ReplyOutput(reply="Year 4, defense... correct?"),
    )
    result = engine.handle_turn(conversation_id, "by doing, 5 hours")
    assert result.reply == "Year 4, defense... correct?"
    assert result.record.phase is Phase.CONFIRMING

    llm.queue(
        turn(Intent.CONFIRM),
        RecommendationOutput(
            recommendations=[Recommendation(name="dueling circle", reason="You love dueling")],
            reply="Try the Dueling Circle!",
        ),
        SummaryOutput(key_points=["Year 4 defense student."]),
    )
    result = engine.handle_turn(conversation_id, "yes")

    record = result.record
    assert result.reply == "Try the Dueling Circle!"
    assert record.phase is Phase.DONE
    assert record.recommendations == [
        Recommendation(name="Dueling Circle", reason="You love dueling")
    ]
    assert record.summary == ["Year 4 defense student."]
    assert record.transcript[-1].text == "Try the Dueling Circle!"
    assert llm.pending == 0


def test_invalid_value_is_kept_out_of_the_profile_and_reported(
    engine: Engine, llm: FakeLLM
) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(turn(school_year=9, specialization="HEALING"))

    record = engine.handle_turn(conversation_id, "year 9, healing").record

    assert record.profile.school_year is None
    assert record.profile.specialization == "HEALING"
    assert "school_year" in record.field_errors

    llm.queue(turn(school_year=7))
    engine.handle_turn(conversation_id, "sorry, 7")
    assert "Rejected values from the last message: school_year" in llm.calls[-1].system


def test_correction_during_confirmation_confirms_again(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = reach_confirmation(engine, llm)
    llm.queue(turn(Intent.CORRECTION, school_year=5), ReplyOutput(reply="Year 5 now. Right?"))

    result = engine.handle_turn(conversation_id, "actually I'm in year 5")

    assert result.reply == "Year 5 now. Right?"
    assert result.record.phase is Phase.CONFIRMING
    assert result.record.profile.school_year == 5


def test_rejection_without_a_fix_goes_back_to_collecting(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = reach_confirmation(engine, llm)
    llm.queue(turn(Intent.REJECT, reply="What should I change?"))

    result = engine.handle_turn(conversation_id, "no, that's wrong")

    assert result.reply == "What should I change?"
    assert result.record.phase is Phase.COLLECTING


def test_optional_field_can_be_skipped(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(turn(Intent.REFUSE, refused=["interests"], school_year=3))

    record = engine.handle_turn(conversation_id, "year 3, rather not share interests").record

    assert record.skipped_fields == ["interests"]


def test_required_field_cannot_be_skipped(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(turn(Intent.REFUSE, refused=["school_year"]))

    record = engine.handle_turn(conversation_id, "not telling").record

    assert record.skipped_fields == []


def test_skipping_the_last_field_moves_to_confirmation(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(turn(**{**FULL_PROFILE, "interests": None}))
    engine.handle_turn(conversation_id, "all but interests")
    llm.queue(turn(Intent.REFUSE, refused=["interests"]), ReplyOutput(reply="Correct?"))

    record = engine.handle_turn(conversation_id, "skip that").record

    assert record.phase is Phase.CONFIRMING


def test_declining_consent_ends_the_conversation(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = engine.start().record.id
    llm.queue(turn(Intent.REJECT, reply="No problem, goodbye!"))

    result = engine.handle_turn(conversation_id, "no thanks")
    assert result.record.phase is Phase.DONE
    assert result.record.summary is None

    later = engine.handle_turn(conversation_id, "hello?")
    assert later.reply == prompts.FINISHED_MESSAGE
    assert len(llm.calls) == 1


def test_stop_mid_interview_finishes_with_a_summary(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(turn(Intent.STOP, reply="Bye!"), SummaryOutput(key_points=["Stopped early."]))

    record = engine.handle_turn(conversation_id, "I want to stop").record

    assert record.phase is Phase.DONE
    assert record.summary == ["Stopped early."]


def test_llm_failure_leaves_the_stored_record_unchanged(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(LLMUnavailableError("timeout"))

    result = engine.handle_turn(conversation_id, "year 3")

    assert result.reply == prompts.LLM_FAILURE_MESSAGE
    assert result.record.version == 2
    assert result.record.transcript[-1].role == "agent"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (LLMRefusalError("no"), prompts.REFUSAL_MESSAGE),
        (LLMOutputError("bad json"), prompts.LLM_FAILURE_MESSAGE),
    ],
)
def test_llm_errors_map_to_fixed_replies(
    engine: Engine, llm: FakeLLM, error: Exception, message: str
) -> None:
    conversation_id = engine.start().record.id
    llm.queue(error)
    assert engine.handle_turn(conversation_id, "hi").reply == message


@pytest.mark.parametrize(
    ("text", "message"),
    [("   ", prompts.EMPTY_INPUT_MESSAGE), ("x" * 2001, prompts.TOO_LONG_MESSAGE)],
)
def test_invalid_input_is_answered_without_calling_the_llm(
    engine: Engine, llm: FakeLLM, text: str, message: str
) -> None:
    conversation_id = engine.start().record.id
    assert engine.handle_turn(conversation_id, text).reply == message
    assert llm.calls == []


def test_unknown_conversation_raises(engine: Engine) -> None:
    with pytest.raises(ConversationNotFoundError):
        engine.handle_turn("0" * 32, "hi")


def test_invented_recommendations_are_dropped(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = reach_confirmation(engine, llm)
    llm.queue(
        turn(Intent.CONFIRM),
        RecommendationOutput(
            recommendations=[
                Recommendation(name="Dueling Circle", reason="duels"),
                Recommendation(name="Battle Alchemy", reason="year 6 only"),
                Recommendation(name="Dragon Riding", reason="does not exist"),
            ],
            reply="Dueling Circle, Battle Alchemy and Dragon Riding!",
        ),
        SummaryOutput(key_points=["s"]),
    )

    result = engine.handle_turn(conversation_id, "yes")

    assert [r.name for r in result.record.recommendations] == ["Dueling Circle"]
    assert "Dragon Riding" not in result.reply
    assert result.reply == prompts.fallback_recommendations(result.record.recommendations)


def test_recommendation_failure_falls_back_to_deterministic_ranking(
    engine: Engine, llm: FakeLLM
) -> None:
    conversation_id = reach_confirmation(engine, llm)
    llm.queue(turn(Intent.CONFIRM), LLMUnavailableError("down"), SummaryOutput(key_points=["s"]))

    record = engine.handle_turn(conversation_id, "yes").record

    assert record.phase is Phase.DONE
    assert record.recommendations[0].name == "Dueling Circle"  # POWER + PRACTICAL


def test_no_eligible_option_is_explained(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(
        turn(**{**FULL_PROFILE, "school_year": 1, "weekly_free_hours": 1}),
        ReplyOutput(reply="Correct?"),
    )
    engine.handle_turn(conversation_id, "year 1, one hour")
    llm.queue(
        turn(Intent.CONFIRM),
        ReplyOutput(reply="Nothing fits yet; more free hours would help."),
        SummaryOutput(key_points=["s"]),
    )

    result = engine.handle_turn(conversation_id, "yes")

    assert result.record.recommendations == []
    assert result.reply == "Nothing fits yet; more free hours would help."
    assert "requires" in llm.calls[-2].messages[0].content


def test_summary_failure_does_not_fail_the_turn(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = reach_confirmation(engine, llm)
    llm.queue(
        turn(Intent.CONFIRM),
        RecommendationOutput(
            recommendations=[Recommendation(name="Dueling Circle", reason="r")], reply="Done!"
        ),
        LLMUnavailableError("down"),
    )

    result = engine.handle_turn(conversation_id, "yes")

    assert result.reply == "Done!"
    assert result.record.phase is Phase.DONE
    assert result.record.summary is None


def test_frustration_changes_the_prompt(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    for _ in range(2):
        llm.queue(turn(sentiment=Sentiment.FRUSTRATED))
        engine.handle_turn(conversation_id, "ugh, this is taking forever")
    llm.queue(turn())
    engine.handle_turn(conversation_id, "fine")

    assert "seems frustrated" in llm.calls[-1].system
    assert "seems frustrated" not in llm.calls[-2].system


def test_language_is_tracked_from_the_student(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = engine.start().record.id
    llm.queue(turn(Intent.CONFIRM, language="es"))
    assert engine.handle_turn(conversation_id, "sí, empecemos").record.language == "es"


def test_history_starts_with_a_user_message_and_alternates(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(turn())
    engine.handle_turn(conversation_id, "year 3")

    roles = [m.role for m in llm.calls[-1].messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


@pytest.mark.parametrize(("code", "expected"), [(" ES ", "es"), ("Spanish", "en"), ("", "en")])
def test_only_iso_language_codes_are_stored(
    engine: Engine, llm: FakeLLM, code: str, expected: str
) -> None:
    conversation_id = engine.start().record.id
    llm.queue(turn(Intent.CONFIRM, language=code))
    assert engine.handle_turn(conversation_id, "yes").record.language == expected


def test_one_off_prompts_tell_the_model_not_to_greet_again() -> None:
    assert "don't greet" in prompts.TASK_SYSTEM_PROMPT


def test_stop_before_consent_ends_without_a_summary(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = engine.start().record.id
    llm.queue(turn(Intent.STOP, reply="Bye!"))

    record = engine.handle_turn(conversation_id, "stop").record

    assert record.phase is Phase.DONE
    assert record.summary is None
    assert len(llm.calls) == 1


def test_off_topic_before_consent_keeps_asking(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = engine.start().record.id
    llm.queue(turn(Intent.OFF_TOPIC, reply="Ha! Shall we begin?"))

    record = engine.handle_turn(conversation_id, "is the food good here?").record

    assert record.phase is Phase.CONSENT


def test_correction_after_recommendations_confirms_again(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = reach_confirmation(engine, llm)
    llm.queue(
        turn(Intent.CONFIRM),
        RecommendationOutput(
            recommendations=[Recommendation(name="Dueling Circle", reason="r")], reply="Done!"
        ),
        SummaryOutput(key_points=["s"]),
    )
    engine.handle_turn(conversation_id, "yes")
    llm.queue(turn(Intent.CORRECTION, school_year=6), ReplyOutput(reply="Year 6 now. Right?"))

    record = engine.handle_turn(conversation_id, "wait, I'm in year 6").record

    assert record.phase is Phase.CONFIRMING
    assert record.profile.school_year == 6


def test_confirmation_failure_uses_the_fallback_message(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(turn(**FULL_PROFILE), LLMUnavailableError("down"))

    result = engine.handle_turn(conversation_id, "everything")

    assert result.record.phase is Phase.CONFIRMING
    assert result.reply == prompts.fallback_confirmation(result.record)
    assert "school year: 4" in result.reply


def test_no_match_failure_uses_the_fallback_message(engine: Engine, llm: FakeLLM) -> None:
    conversation_id = start_collecting(engine, llm)
    llm.queue(
        turn(**{**FULL_PROFILE, "school_year": 1, "weekly_free_hours": 0}),
        ReplyOutput(reply="Correct?"),
    )
    engine.handle_turn(conversation_id, "year 1, no free time")
    llm.queue(turn(Intent.CONFIRM), LLMOutputError("bad"), SummaryOutput(key_points=["s"]))

    result = engine.handle_turn(conversation_id, "yes")

    assert result.reply == prompts.fallback_recommendations([])
    assert result.record.phase is Phase.DONE


def test_concurrent_turns_on_the_same_conversation_conflict(
    engine: Engine, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = engine.start().record
    llm.queue(turn(Intent.CONFIRM))
    engine.handle_turn(stale.id, "yes")  # another request moved it to version 2

    # The engine reads the stale copy; save() then checks against the real stored version.
    repository = engine._repository
    real_get = repository.get
    reads = iter([stale])
    monkeypatch.setattr(repository, "get", lambda cid: next(reads, None) or real_get(cid))
    llm.queue(turn(Intent.CONFIRM))

    with pytest.raises(VersionConflictError):
        engine.handle_turn(stale.id, "yes again")
