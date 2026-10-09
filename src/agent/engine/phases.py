"""Interview flow rules: which fields are pending, and what each turn leads to.

``next_action`` is the whole phase-transition table as one pure function, so the flow can be
read in one place and tested without an LLM.
"""

from collections.abc import Sequence
from enum import StrEnum

from agent.domain.models import (
    ConversationRecord,
    Intent,
    Phase,
    ProfileField,
    Sentiment,
    StudentProfile,
)


class Action(StrEnum):
    REPLY = "reply"  # stay in the phase; send the LLM's reply
    START_COLLECTING = "start_collecting"  # consent given
    BACK_TO_COLLECTING = "back_to_collecting"  # the summary was wrong, no fix given yet
    CONFIRM = "confirm"  # show the profile and ask if it's right
    RECOMMEND = "recommend"  # profile confirmed
    DECLINE = "decline"  # no consent: end without an interview
    STOP = "stop"  # the student ends the interview early


def next_action(phase: Phase, intent: Intent, *, changed: bool, ready: bool) -> Action:
    """What a turn leads to.

    ``changed``: the profile changed this turn (or a field was skipped).
    ``ready``: nothing is pending and no value was rejected.
    """
    if intent is Intent.STOP:
        match phase:
            case Phase.CONSENT:
                return Action.DECLINE
            case Phase.DONE:
                return Action.REPLY
            case _:
                return Action.STOP

    ready_to_confirm = ready and (changed or intent is Intent.CONFIRM)
    match phase:
        case Phase.CONSENT:
            if intent in (Intent.REJECT, Intent.REFUSE):
                return Action.DECLINE
            if intent is Intent.OFF_TOPIC:
                return Action.REPLY
            return Action.CONFIRM if ready_to_confirm else Action.START_COLLECTING
        case Phase.COLLECTING:
            return Action.CONFIRM if ready_to_confirm else Action.REPLY
        case Phase.CONFIRMING:
            if changed:
                return Action.CONFIRM
            if intent is Intent.CONFIRM:
                return Action.RECOMMEND
            if intent in (Intent.REJECT, Intent.CORRECTION):
                return Action.BACK_TO_COLLECTING
            return Action.REPLY
        case Phase.DONE:
            return Action.CONFIRM if changed and ready else Action.REPLY


def pending_fields(record: ConversationRecord) -> list[ProfileField]:
    """Fields still to ask about, in the order the bot should ask them."""
    return [f for f in record.profile.missing_fields() if f not in record.skipped_fields]


def ready_for_confirmation(record: ConversationRecord) -> bool:
    return not pending_fields(record) and not record.field_errors


def skip_fields(record: ConversationRecord, fields: Sequence[ProfileField]) -> list[ProfileField]:
    """Mark the fields the student declined to answer as skipped, where that is allowed.

    Returns the fields that were skipped; required fields are never skipped.
    """
    pending = pending_fields(record)
    skippable = StudentProfile.skippable_fields()
    skipped = [f for f in fields if f in skippable and f in pending]
    record.skipped_fields.extend(skipped)
    return skipped


def is_frustrated(record: ConversationRecord, streak: int) -> bool:
    """True after ``streak`` frustrated messages in a row."""
    recent = record.sentiment_history[-streak:]
    return len(recent) == streak and all(s is Sentiment.FRUSTRATED for s in recent)
