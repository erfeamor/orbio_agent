"""Key points of a finished conversation, for school staff."""

import logging

from agent.domain.models import ConversationRecord
from agent.engine import prompts
from agent.engine.schemas import SummaryOutput
from agent.llm.base import LLMClient, RecoverableLLMError

logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def summarize(self, record: ConversationRecord) -> list[str] | None:
        """The key points, or None if the LLM failed: the summary must never fail a turn."""
        try:
            return self._llm.generate(
                system=prompts.TASK_SYSTEM_PROMPT,
                messages=prompts.summary_request(record),
                output_type=SummaryOutput,
            ).key_points
        except RecoverableLLMError:
            logger.exception("summary failed for conversation %s", record.id)
            return None
