"""Which fields are required, and when the interview is ready to move on."""

from agent.domain.models import ConversationRecord, Sentiment

# Needed by the catalog's hard filters, so the interview cannot finish without them.
REQUIRED_FIELDS = ("school_year", "specialization", "weekly_free_hours")
# Only used to rank recommendations; the student may decline to answer.
SKIPPABLE_FIELDS = ("interests", "magic_orientation", "learning_modality")

FRUSTRATION_STREAK = 2


def pending_fields(record: ConversationRecord) -> list[str]:
    """Fields still to ask about, in the order the bot should ask them."""
    return [f for f in record.profile.missing_fields() if f not in record.skipped_fields]


def ready_for_confirmation(record: ConversationRecord) -> bool:
    return not pending_fields(record) and not record.field_errors


def skip_fields(record: ConversationRecord, fields: list[str]) -> list[str]:
    """Mark the fields the student declined to answer as skipped, where that is allowed.

    Returns the fields that were skipped; required fields are never skipped.
    """
    skipped = [f for f in fields if f in SKIPPABLE_FIELDS and f in pending_fields(record)]
    record.skipped_fields.extend(skipped)
    return skipped


def is_frustrated(record: ConversationRecord) -> bool:
    recent = record.sentiment_history[-FRUSTRATION_STREAK:]
    return len(recent) == FRUSTRATION_STREAK and all(s is Sentiment.FRUSTRATED for s in recent)
