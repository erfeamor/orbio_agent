"""The interview engine: one student message in, one reply out.

The LLM extracts information and writes replies; this module decides the phase transitions,
so the flow stays predictable and testable.
"""

import logging
import re
from dataclasses import dataclass

from agent.domain.catalog import CatalogEntry, filter_catalog, rank_by_ideal_profile
from agent.domain.models import (
    ConversationRecord,
    Intent,
    Phase,
    RecommendationOutput,
    ReplyOutput,
    SummaryOutput,
    TurnOutput,
)
from agent.engine import phases, prompts
from agent.llm.base import LLMClient, LLMRefusalError, RecoverableLLMError
from agent.storage.base import ConversationRepository

logger = logging.getLogger(__name__)

_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")

MAX_RECOMMENDATIONS = 3


class ConversationNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class TurnResult:
    reply: str
    record: ConversationRecord


class Engine:
    def __init__(
        self,
        llm: LLMClient,
        repository: ConversationRepository,
        catalog: list[CatalogEntry],
    ) -> None:
        self._llm = llm
        self._repository = repository
        self._catalog = catalog

    def start(self) -> TurnResult:
        record = ConversationRecord(
            model=self._llm.model_name, prompt_version=prompts.PROMPT_VERSION
        )
        record.add_message("agent", prompts.CONSENT_MESSAGE)
        return TurnResult(prompts.CONSENT_MESSAGE, self._repository.save(record))

    def handle_turn(self, conversation_id: str, text: str) -> TurnResult:
        record = self._repository.get(conversation_id)
        if record is None:
            raise ConversationNotFoundError(conversation_id)

        text = text.strip()
        if not text:
            return TurnResult(prompts.EMPTY_INPUT_MESSAGE, record)
        if len(text) > prompts.MAX_INPUT_CHARS:
            return TurnResult(prompts.TOO_LONG_MESSAGE, record)
        if record.phase is Phase.DONE and record.summary is None and not record.recommendations:
            # Declined consent or stopped early: nothing more to talk about.
            return TurnResult(prompts.FINISHED_MESSAGE, record)

        # Work on a copy: if the turn fails, the stored record is left exactly as it was,
        # so the student can simply repeat their message.
        working = record.model_copy(deep=True)
        working.add_message("student", text)
        try:
            reply = self._advance(working)
        except LLMRefusalError:
            logger.warning("turn refused in conversation %s", record.id)
            return TurnResult(prompts.REFUSAL_MESSAGE, record)
        except RecoverableLLMError:
            logger.exception("turn failed in conversation %s", record.id)
            return TurnResult(prompts.LLM_FAILURE_MESSAGE, record)

        working.add_message("agent", reply)
        return TurnResult(reply, self._repository.save(working))

    def _advance(self, record: ConversationRecord) -> str:
        """Run one interview turn on ``record`` (mutated in place) and return the reply."""
        turn = self._llm.generate(
            system=prompts.interview_system_prompt(record),
            messages=prompts.history(record),
            output_type=TurnOutput,
        )
        record.language = _language_code(turn.language) or record.language
        record.sentiment_history.append(turn.sentiment)

        previous = record.profile
        record.profile, record.field_errors = record.profile.apply(turn.extracted.model_dump())
        changed = record.profile != previous
        # A field the student answers later is no longer skipped.
        record.skipped_fields = [
            f for f in record.skipped_fields if getattr(record.profile, f) is None
        ]

        if turn.intent is Intent.STOP:
            if record.phase is Phase.CONSENT:
                record.phase = Phase.DONE
                return turn.reply
            return self._finish(record, turn.reply)

        match record.phase:
            case Phase.CONSENT:
                if turn.intent in (Intent.REJECT, Intent.REFUSE):
                    record.phase = Phase.DONE
                    return turn.reply
                if turn.intent is Intent.OFF_TOPIC:
                    return turn.reply
                record.phase = Phase.COLLECTING

            case Phase.COLLECTING:
                skipped = phases.skip_fields(record, turn.refused_fields)
                changed = changed or bool(skipped)

            case Phase.CONFIRMING:
                if changed:
                    return self._confirm(record)
                if turn.intent is Intent.CONFIRM:
                    return self._recommend(record)
                if turn.intent in (Intent.REJECT, Intent.CORRECTION):
                    record.phase = Phase.COLLECTING
                return turn.reply

            case Phase.DONE:
                if changed and phases.ready_for_confirmation(record):
                    return self._confirm(record)
                return turn.reply

        if phases.ready_for_confirmation(record) and (changed or turn.intent is Intent.CONFIRM):
            return self._confirm(record)
        return turn.reply

    def _confirm(self, record: ConversationRecord) -> str:
        record.phase = Phase.CONFIRMING
        try:
            return self._llm.generate(
                system=prompts.TASK_SYSTEM_PROMPT,
                messages=prompts.confirmation_request(record),
                output_type=ReplyOutput,
            ).reply
        except RecoverableLLMError:
            logger.exception("confirmation message failed; using fallback")
            return prompts.fallback_confirmation(record)

    def _recommend(self, record: ConversationRecord) -> str:
        result = filter_catalog(self._catalog, record.profile)
        if not result.eligible:
            record.recommendations = []
            try:
                reply = self._llm.generate(
                    system=prompts.TASK_SYSTEM_PROMPT,
                    messages=prompts.no_match_request(record, result.rejected),
                    output_type=ReplyOutput,
                ).reply
            except RecoverableLLMError:
                logger.exception("no-match message failed; using fallback")
                reply = prompts.fallback_recommendations([])
            return self._finish(record, reply)

        try:
            output = self._llm.generate(
                system=prompts.TASK_SYSTEM_PROMPT,
                messages=prompts.recommendation_request(
                    record, result.eligible, MAX_RECOMMENDATIONS
                ),
                output_type=RecommendationOutput,
            )
        except RecoverableLLMError:
            logger.exception("recommendation call failed; using deterministic ranking")
            output = None

        eligible_names = {entry.name.casefold(): entry.name for entry in result.eligible}
        valid = []
        if output is not None:
            for rec in output.recommendations[:MAX_RECOMMENDATIONS]:
                name = eligible_names.get(rec.name.strip().casefold())
                if name is None:
                    logger.warning("dropping recommendation not in the catalog: %r", rec.name)
                    continue
                valid.append(rec.model_copy(update={"name": name}))

        if output is not None and valid and len(valid) == len(output.recommendations):
            reply = output.reply
        else:
            # The LLM failed or named something we can't vouch for: don't repeat its reply.
            valid = valid or rank_by_ideal_profile(
                result.eligible, record.profile, MAX_RECOMMENDATIONS
            )
            reply = prompts.fallback_recommendations(valid)

        record.recommendations = valid
        return self._finish(record, reply)

    def _finish(self, record: ConversationRecord, reply: str) -> str:
        record.phase = Phase.DONE
        try:
            record.summary = self._llm.generate(
                system=prompts.TASK_SYSTEM_PROMPT,
                messages=prompts.summary_request(record),
                output_type=SummaryOutput,
            ).key_points
        except RecoverableLLMError:
            # The summary is for staff; failing to write it must not fail the student's turn.
            logger.exception("summary failed for conversation %s", record.id)
        return reply


def _language_code(value: str) -> str | None:
    """Accept only ISO 639-1 codes ("es"); anything else keeps the previous language."""
    code = value.strip().lower()
    if _LANGUAGE_CODE.fullmatch(code):
        return code
    logger.warning("ignoring invalid language code %r", value)
    return None
