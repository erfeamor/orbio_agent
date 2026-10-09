"""Builders shared by the tests (pytest puts this directory on the import path)."""

from pathlib import Path
from typing import Any

from agent.domain.catalog import load_catalog
from agent.domain.models import Intent, ProfileField, Sentiment
from agent.engine.engine import Engine, EngineConfig
from agent.engine.messages import MESSAGES
from agent.engine.recommender import Recommender
from agent.engine.schemas import ExtractedFields, TurnOutput
from agent.engine.summarizer import Summarizer
from agent.llm.base import LLMClient
from agent.llm.fake import FakeCall
from agent.storage.base import ConversationRepository

CATALOG = load_catalog(Path(__file__).parents[1] / "data" / "catalog.json")
EN = MESSAGES["en"]


def turn(
    intent: Intent = Intent.ANSWER,
    reply: str = "llm reply",
    sentiment: Sentiment = Sentiment.NEUTRAL,
    language: str = "en",
    refused: list[ProfileField] | None = None,
    **extracted: Any,
) -> TurnOutput:
    """A scripted interview-turn output from the LLM."""
    return TurnOutput(
        extracted=ExtractedFields(**extracted),
        reply=reply,
        intent=intent,
        sentiment=sentiment,
        language=language,
        refused_fields=refused or [],
    )


def make_engine(
    llm: LLMClient,
    repository: ConversationRepository,
    *,
    max_input_chars: int = 2000,
    frustration_streak: int = 2,
    max_recommendations: int = 3,
) -> Engine:
    return Engine(
        llm,
        repository,
        Recommender(llm, CATALOG, limit=max_recommendations),
        Summarizer(llm),
        EngineConfig(max_input_chars=max_input_chars, frustration_streak=frustration_streak),
    )


def sent_state(call: FakeCall) -> str:
    """What an interview call sent with the student's message: state plus the message itself."""
    return call.messages[-1].content
