"""A full scripted conversation, checked against the JSON file written to disk."""

import json
from pathlib import Path
from typing import Any

from agent.domain.catalog import load_catalog
from agent.domain.models import (
    ExtractedFields,
    Intent,
    Recommendation,
    RecommendationOutput,
    ReplyOutput,
    Sentiment,
    SummaryOutput,
    TurnOutput,
)
from agent.engine.engine import Engine
from agent.llm.fake import FakeLLM
from agent.storage.file_repo import JsonFileRepository

CATALOG = load_catalog(Path(__file__).parents[1] / "data" / "catalog.json")


def turn(
    intent: Intent = Intent.ANSWER,
    sentiment: Sentiment = Sentiment.NEUTRAL,
    refused: list[str] | None = None,
    **extracted: Any,
) -> TurnOutput:
    return TurnOutput(
        extracted=ExtractedFields(**extracted),
        reply=f"respuesta ({intent.value})",
        intent=intent,
        sentiment=sentiment,
        language="es",
        refused_fields=refused or [],
    )


def test_spanish_conversation_with_errors_skip_and_correction(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = Engine(llm, JsonFileRepository(tmp_path), CATALOG)
    conversation_id = engine.start().record.id

    script: list[tuple[str, list[Any]]] = [
        ("sí, empecemos", [turn(Intent.CONFIRM)]),
        ("año 9, alquimia", [turn(school_year=9, specialization="alchemy")]),
        ("perdón, año 3", [turn(Intent.CORRECTION, school_year=3)]),
        (
            "prefiero no decir mis intereses, ¡qué pesado!",
            [turn(Intent.REFUSE, Sentiment.FRUSTRATED, refused=["interests"])],
        ),
        (
            "conocimiento, en grupo",
            [
                turn(
                    sentiment=Sentiment.FRUSTRATED,
                    magic_orientation="KNOWLEDGE",
                    learning_modality="COLLABORATIVE",
                )
            ],
        ),
        (
            "4 horas",
            [turn(weekly_free_hours=4), ReplyOutput(reply="¿Año 3, alquimia, 4 horas?")],
        ),
        (
            "no, son 5 horas",
            [turn(Intent.CORRECTION, weekly_free_hours=5), ReplyOutput(reply="¿Ahora 5 horas?")],
        ),
        (
            "sí",
            [
                turn(Intent.CONFIRM, Sentiment.POSITIVE),
                RecommendationOutput(
                    recommendations=[
                        Recommendation(name="Ancient Runes Study Group", reason="Conocimiento"),
                        Recommendation(name="Potion Brewers Guild", reason="Alquimia"),
                    ],
                    reply="Te recomiendo las Runas Antiguas y el Gremio de Pociones.",
                ),
                SummaryOutput(key_points=["Third-year alchemist.", "Frustrated mid-interview."]),
            ],
        ),
        ("¿qué hace el gremio?", [turn(Intent.OFF_TOPIC)]),
    ]
    for text, responses in script:
        llm.queue(*responses)
        engine.handle_turn(conversation_id, text)
    assert llm.pending == 0

    stored = json.loads((tmp_path / f"{conversation_id}.json").read_text(encoding="utf-8"))

    assert stored["phase"] == "done"
    assert stored["language"] == "es"
    assert stored["version"] == len(script) + 1
    assert stored["model"] == "fake"
    assert stored["prompt_version"] == "v2"
    assert stored["profile"] == {
        "school_year": 3,
        "specialization": "ALCHEMY",
        "interests": None,
        "magic_orientation": "KNOWLEDGE",
        "learning_modality": "COLLABORATIVE",
        "weekly_free_hours": 5.0,
    }
    assert stored["skipped_fields"] == ["interests"]
    assert stored["field_errors"] == {}
    assert stored["sentiment_history"] == [
        "neutral",
        "neutral",
        "neutral",
        "frustrated",
        "frustrated",
        "neutral",
        "neutral",
        "positive",
        "neutral",
    ]
    assert stored["recommendations"] == [
        {"name": "Ancient Runes Study Group", "reason": "Conocimiento"},
        {"name": "Potion Brewers Guild", "reason": "Alquimia"},
    ]
    assert stored["summary"] == ["Third-year alchemist.", "Frustrated mid-interview."]

    roles = [m["role"] for m in stored["transcript"]]
    assert roles == ["agent"] + ["student", "agent"] * len(script)
    assert all(m["timestamp"] for m in stored["transcript"])

    # The rejected year 9 was reported back in the next prompt, and frustration
    # switched the prompt to its short mode after two frustrated messages.
    assert "school_year:" in llm.calls[2].system
    assert "seems frustrated" in llm.calls[5].system
