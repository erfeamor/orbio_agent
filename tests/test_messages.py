import pytest
from factories import CATALOG

from agent.domain.catalog import RankedEntry, rank_by_ideal_profile
from agent.domain.models import Recommendation, StudentProfile
from agent.engine.messages import MESSAGES, messages_for


def test_unknown_or_missing_language_falls_back_to_english() -> None:
    assert messages_for("fr") is MESSAGES["en"]
    assert messages_for(None) is MESSAGES["en"]
    assert messages_for("es") is MESSAGES["es"]


@pytest.mark.parametrize("language", sorted(MESSAGES))
def test_every_language_labels_every_profile_field(language: str) -> None:
    assert set(MESSAGES[language].field_labels) == set(StudentProfile.model_fields)


def test_fallback_confirmation_uses_localized_labels() -> None:
    profile = StudentProfile(school_year=3, weekly_free_hours=4)
    text = MESSAGES["es"].fallback_confirmation(profile)
    assert text == "Esto es lo que tengo: curso: 3, horas libres por semana: 4.0. ¿Es correcto?"


def test_fallback_recommendations() -> None:
    es = MESSAGES["es"]
    assert es.fallback_recommendations([]) == es.no_match
    text = es.fallback_recommendations([Recommendation(name="Dueling Circle", reason="r")])
    assert text.startswith("Te recomiendo: Dueling Circle: r")


def test_ranking_reports_matches_and_orders_by_score() -> None:
    profile = StudentProfile.model_validate(
        {"magic_orientation": "POWER", "learning_modality": "PRACTICAL"}
    )
    ranked = rank_by_ideal_profile(CATALOG, profile, limit=3)

    assert len(ranked) == 3
    assert ranked[0].score == 2  # Dueling Circle / Battle Alchemy: POWER + PRACTICAL
    assert [r.score for r in ranked] == sorted((r.score for r in ranked), reverse=True)


def test_match_reason_is_worded_per_language() -> None:
    entry = next(e for e in CATALOG if e.name == "Dueling Circle")
    both = RankedEntry(entry, matches_orientation=True, matches_modality=True)
    none = RankedEntry(entry, matches_orientation=False, matches_modality=False)

    assert MESSAGES["en"].match_reason(both) == (
        "suits your POWER drive and fits your PRACTICAL learning style"
    )
    assert " y " in MESSAGES["es"].match_reason(both)
    assert MESSAGES["en"].match_reason(none) == "open to you"
