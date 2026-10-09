"""Recommendations: hard filters in Python, ranking by the LLM, names checked against the catalog.

This is the retrieval step of the bonus RAG feature: the catalog is filtered on metadata, and
only eligible entries reach the LLM, which must answer with exact catalog names.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from agent.domain.catalog import (
    CatalogEntry,
    Rejection,
    filter_catalog,
    rank_by_ideal_profile,
)
from agent.domain.models import ConversationRecord, Recommendation
from agent.engine import prompts
from agent.engine.messages import messages_for
from agent.engine.schemas import RecommendationOutput, ReplyOutput
from agent.llm.base import LLMClient, RecoverableLLMError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: list[Recommendation]
    reply: str


class Recommender:
    def __init__(self, llm: LLMClient, catalog: list[CatalogEntry], *, limit: int) -> None:
        self._llm = llm
        self._catalog = catalog
        self._by_name = {entry.name: entry for entry in catalog}
        self._limit = limit

    @property
    def catalog_by_name(self) -> Mapping[str, CatalogEntry]:
        return self._by_name

    def recommend(self, record: ConversationRecord) -> RecommendationResult:
        result = filter_catalog(self._catalog, record.profile)
        if not result.eligible:
            return RecommendationResult([], self._explain_no_match(record, result.rejected))

        try:
            output: RecommendationOutput | None = self._llm.generate(
                system=prompts.TASK_SYSTEM_PROMPT,
                messages=prompts.recommendation_request(record, result.eligible, self._limit),
                output_type=RecommendationOutput,
            )
        except RecoverableLLMError:
            logger.exception("recommendation call failed; using deterministic ranking")
            output = None

        valid = self._known_names(output, result.eligible)
        if output is not None and valid and len(valid) == len(output.recommendations):
            return RecommendationResult(valid, output.reply)

        # The LLM failed or named something we can't vouch for: don't repeat its reply.
        messages = messages_for(record.language)
        valid = valid or [
            Recommendation(name=ranked.entry.name, reason=messages.match_reason(ranked))
            for ranked in rank_by_ideal_profile(result.eligible, record.profile, self._limit)
        ]
        return RecommendationResult(valid, messages.fallback_recommendations(valid))

    def _known_names(
        self, output: RecommendationOutput | None, eligible: list[CatalogEntry]
    ) -> list[Recommendation]:
        """The LLM's picks that are eligible catalog entries, with the catalog's exact name."""
        if output is None:
            return []
        names = {entry.name.casefold(): entry.name for entry in eligible}
        valid = []
        for rec in output.recommendations[: self._limit]:
            name = names.get(rec.name.strip().casefold())
            if name is None:
                logger.warning("dropping recommendation not in the catalog: %r", rec.name)
                continue
            valid.append(rec.model_copy(update={"name": name}))
        return valid

    def _explain_no_match(
        self, record: ConversationRecord, rejected: Mapping[str, list[Rejection]]
    ) -> str:
        try:
            return self._llm.generate(
                system=prompts.TASK_SYSTEM_PROMPT,
                messages=prompts.no_match_request(record, rejected),
                output_type=ReplyOutput,
            ).reply
        except RecoverableLLMError:
            logger.exception("no-match message failed; using fallback")
            return messages_for(record.language).fallback_recommendations([])
