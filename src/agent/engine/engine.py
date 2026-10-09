"""The interview engine: one student message in, one reply out.

The LLM extracts information and writes replies; ``phases.next_action`` decides what each turn
leads to, so the flow stays predictable and testable. Recommending and summarizing are
delegated to ``Recommender`` and ``Summarizer``.
"""

import logging
import re
from dataclasses import dataclass
from typing import assert_never

from agent.domain.models import ConversationRecord, EndReason, Phase, Role
from agent.engine import phases, prompts
from agent.engine.messages import messages_for
from agent.engine.phases import Action
from agent.engine.recommender import Recommender
from agent.engine.schemas import ReplyOutput, TurnOutput
from agent.engine.summarizer import Summarizer
from agent.llm.base import LLMClient, LLMRefusalError, RecoverableLLMError
from agent.logging_config import bound, conversation_id_var
from agent.storage.base import ConversationRepository

logger = logging.getLogger(__name__)

_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")


class ConversationNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class EngineConfig:
    max_input_chars: int
    frustration_streak: int


@dataclass(frozen=True)
class TurnResult:
    reply: str
    record: ConversationRecord


class Engine:
    def __init__(
        self,
        llm: LLMClient,
        repository: ConversationRepository,
        recommender: Recommender,
        summarizer: Summarizer,
        config: EngineConfig,
    ) -> None:
        self._llm = llm
        self._repository = repository
        self._recommender = recommender
        self._summarizer = summarizer
        self._config = config

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def start(self, language: str | None = None) -> TurnResult:
        """Open a conversation; ``language`` (e.g. the browser's) picks the consent text."""
        record = ConversationRecord(
            model=self._llm.model_name, prompt_version=prompts.PROMPT_VERSION
        )
        if language and (code := _language_code(language)):
            record.language = code
        consent = messages_for(record.language).consent
        record.add_message(Role.AGENT, consent)
        return TurnResult(consent, self._repository.save(record))

    def get(self, conversation_id: str) -> ConversationRecord | None:
        return self._repository.get(conversation_id)

    def handle_turn(self, conversation_id: str, text: str) -> TurnResult:
        # Every log line written during the turn carries the conversation id.
        with bound(conversation_id_var, conversation_id):
            return self._handle_turn(conversation_id, text)

    def _handle_turn(self, conversation_id: str, text: str) -> TurnResult:
        record = self._repository.get(conversation_id)
        if record is None:
            raise ConversationNotFoundError(conversation_id)

        messages = messages_for(record.language)
        text = text.strip()
        if not text:
            return TurnResult(messages.empty_input, record)
        if len(text) > self._config.max_input_chars:
            return TurnResult(messages.too_long, record)
        if record.end_reason is EndReason.DECLINED:
            # No interview took place, so there is nothing to follow up on.
            return TurnResult(messages.finished, record)

        # Work on a copy: if the turn fails, the stored record is left exactly as it was,
        # so the student can simply repeat their message.
        working = record.model_copy(deep=True)
        working.add_message(Role.STUDENT, text)
        try:
            reply = self._advance(working)
        except LLMRefusalError:
            logger.warning("turn refused in conversation %s", record.id)
            return TurnResult(messages.refusal, record)
        except RecoverableLLMError:
            logger.exception("turn failed in conversation %s", record.id)
            return TurnResult(messages.llm_failure, record)

        working.add_message(Role.AGENT, reply)
        just_ended = working.phase is Phase.DONE and record.phase is not Phase.DONE
        if just_ended and working.end_reason is not EndReason.DECLINED:
            # Written after the final reply is in the transcript, once per ending.
            working.summary = self._summarizer.summarize(working)
        return TurnResult(reply, self._repository.save(working))

    def _advance(self, record: ConversationRecord) -> str:
        """Run one interview turn on ``record`` (mutated in place) and return the reply."""
        state = prompts.interview_state(
            record,
            self._recommender.catalog_by_name,
            frustrated=phases.is_frustrated(record, self._config.frustration_streak),
        )
        turn = self._llm.generate(
            system=prompts.INTERVIEW_SYSTEM_PROMPT,
            messages=prompts.interview_messages(record, state),
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
        if record.phase is Phase.COLLECTING:
            changed = bool(phases.skip_fields(record, turn.refused_fields)) or changed

        action = phases.next_action(
            record.phase,
            turn.intent,
            changed=changed,
            ready=phases.ready_for_confirmation(record),
        )
        return self._perform(action, record, turn.reply)

    def _perform(self, action: Action, record: ConversationRecord, reply: str) -> str:
        match action:
            case Action.REPLY:
                return reply
            case Action.START_COLLECTING | Action.BACK_TO_COLLECTING:
                record.phase = Phase.COLLECTING
                return reply
            case Action.CONFIRM:
                return self._confirm(record)
            case Action.RECOMMEND:
                result = self._recommender.recommend(record)
                record.recommendations = result.recommendations
                return self._end(record, EndReason.COMPLETED, result.reply)
            case Action.DECLINE:
                return self._end(record, EndReason.DECLINED, reply)
            case Action.STOP:
                return self._end(record, EndReason.STOPPED, reply)
            case unreachable:
                assert_never(unreachable)

    def _confirm(self, record: ConversationRecord) -> str:
        record.phase = Phase.CONFIRMING
        record.end_reason = None  # a correction after the end reopens the conversation
        try:
            return self._llm.generate(
                system=prompts.TASK_SYSTEM_PROMPT,
                messages=prompts.confirmation_request(record),
                output_type=ReplyOutput,
            ).reply
        except RecoverableLLMError:
            logger.exception("confirmation message failed; using fallback")
            return messages_for(record.language).fallback_confirmation(record.profile)

    @staticmethod
    def _end(record: ConversationRecord, reason: EndReason, reply: str) -> str:
        record.phase = Phase.DONE
        record.end_reason = reason
        return reply


def _language_code(value: str) -> str | None:
    """Accept ISO 639-1 codes ("es", or a tag like "es-ES"); anything else is ignored."""
    code = value.strip().lower().split("-")[0]
    if _LANGUAGE_CODE.fullmatch(code):
        return code
    logger.warning("ignoring invalid language code %r", value)
    return None
