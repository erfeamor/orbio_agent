"""Prompts for every LLM call, plus the fixed messages used when no LLM call is possible."""

import json

from agent.domain.catalog import CatalogEntry
from agent.domain.models import (
    ConversationRecord,
    LearningModality,
    MagicOrientation,
    Phase,
    Recommendation,
    Specialization,
    StudentProfile,
)
from agent.engine import phases
from agent.llm.base import ChatMessage

PROMPT_VERSION = "v2"

# --- Fixed messages (English; used at the start and when the LLM is unavailable) ---

CONSENT_MESSAGE = (
    "Welcome to the Arcanum Academy advisor! I'll ask a few quick questions about your school "
    "year, specialization, interests, what drives your magic, how you like to learn and your free "
    "time, so I can recommend clubs and advanced classes. Your answers are saved so the school "
    "can follow up. Shall we begin?"
)
EMPTY_INPUT_MESSAGE = "Sorry, I didn't catch that. Could you say it again?"
TOO_LONG_MESSAGE = "That's a bit long for me. Could you answer in a shorter message?"
LLM_FAILURE_MESSAGE = "Sorry, I had trouble thinking just now. Could you repeat that?"
REFUSAL_MESSAGE = (
    "I can't help with that. Let's get back to finding you the right clubs and classes."
)
FINISHED_MESSAGE = "This conversation has ended. Start a new one any time!"

MAX_INPUT_CHARS = 2000


def _options(enum: type[Specialization | MagicOrientation | LearningModality]) -> str:
    return " | ".join(member.value for member in enum)


INTERVIEW_SYSTEM_PROMPT = f"""\
You are the student advisor of the Arcanum Academy, a school of magic. You run a short, friendly \
interview so the school can recommend clubs and advanced classes to the student.

## What to collect, in this order
- school_year: integer from 1 to 7
- specialization: {_options(Specialization)}
- interests: free text, what the student enjoys or wants to explore
- magic_orientation: what drives their magic: {_options(MagicOrientation)}
- learning_modality: how they learn best: {_options(LearningModality)}
- weekly_free_hours: hours per week they could spend on clubs or classes (0 to 168)

## How to talk
- Your reply is read aloud. Keep it under 50 words, warm and natural. No lists, markdown or emojis.
- Ask about one or two missing fields at a time, in the order above. Don't present the options \
as a menu unless the student is unsure; describe them in plain words.
- Always reply in the student's language.
- Never recommend clubs or classes yourself and never invent facts about the school; \
recommendations come after the student confirms their profile.
- If the student goes off topic, answer in one short sentence and steer back to the interview.

## How to fill `extracted`
- Only fields the student stated in their latest message; leave everything else null.
- Map clear descriptions to an option ("I learn by doing" -> PRACTICAL, "I want to help \
people" -> ALTRUISM). If they name something that is not an option ("necromancy"), copy their \
word as-is so it can be rejected; never pick an option they didn't express.
- Numbers as numbers: "third year" -> 3, "about five hours" -> 5.

## Labels
- intent: answer (gives information), correction (changes something said before), confirm \
(agrees / says yes), reject (says no or that something is wrong, without the fix), \
off_topic, refuse (declines to answer a question), stop (wants to end the conversation).
- refused_fields: the fields they declined to answer, if intent is refuse.
- sentiment: frustrated if they show impatience or annoyance, positive if enthusiastic, \
otherwise neutral.
- language: ISO 639-1 code of the language the student wrote in.
"""

_PHASE_INSTRUCTIONS = {
    Phase.CONSENT: (
        "The student has just seen a consent message asking whether to begin. If they agree, "
        "thank them and ask the first question. If they decline, say a kind goodbye. "
        "If unclear, briefly ask again whether they'd like to start."
    ),
    Phase.COLLECTING: (
        "Collect the pending fields. If a value was rejected, explain why in a few words and ask "
        f"again. The student may skip {', '.join(phases.SKIPPABLE_FIELDS)}: acknowledge and move "
        f"on. {', '.join(phases.REQUIRED_FIELDS)} are needed to check eligibility: if refused, "
        "explain that briefly and offer to end the interview instead."
    ),
    Phase.CONFIRMING: (
        "You showed the student a summary of their profile and asked whether it is correct. "
        "If they say yes, intent is confirm (your reply can just be a short acknowledgement). "
        "If something is wrong but they didn't say the correct value, ask what to change."
    ),
    Phase.DONE: (
        "The interview is over and the recommendations below were given. Answer short follow-up "
        "questions using only those recommendations. If the student corrects their profile, "
        "extract the new value."
    ),
}


def _known_values(profile: StudentProfile) -> str:
    return json.dumps(profile.model_dump(mode="json", exclude_none=True), ensure_ascii=False)


def interview_system_prompt(record: ConversationRecord) -> str:
    """Static instructions followed by the current state of the interview."""
    state = [
        "## Current state",
        f"Phase: {record.phase.value}",
        f"Known values: {_known_values(record.profile)}",
        f"Pending fields: {', '.join(phases.pending_fields(record)) or 'none'}",
    ]
    if record.skipped_fields:
        state.append(f"Skipped by the student: {', '.join(record.skipped_fields)}")
    if record.field_errors:
        errors = "; ".join(f"{k}: {v}" for k, v in record.field_errors.items())
        state.append(f"Rejected values from the last message: {errors}")
    if record.recommendations:
        recs = "; ".join(f"{r.name} ({r.reason})" for r in record.recommendations)
        state.append(f"Recommendations given: {recs}")
    if phases.is_frustrated(record):
        state.append(
            "The student seems frustrated: be extra brief, acknowledge it once, and offer to skip "
            "optional questions or finish early."
        )
    state.append(f"Instructions for this phase: {_PHASE_INSTRUCTIONS[record.phase]}")
    return INTERVIEW_SYSTEM_PROMPT + "\n" + "\n".join(state)


def history(record: ConversationRecord) -> list[ChatMessage]:
    """The transcript as chat messages. The API expects the first message from the user."""
    messages = [ChatMessage("user", "[The student opened the advisor chat]")]
    messages += [
        ChatMessage("user" if m.role == "student" else "assistant", m.text)
        for m in record.transcript
    ]
    return messages


# --- One-off calls ---

TASK_SYSTEM_PROMPT = (
    "You are the student advisor of the Arcanum Academy, a school of magic, in the middle of an "
    "interview with a student: don't greet them or introduce yourself again. Your replies are "
    "read aloud: warm, natural, no lists, markdown or emojis."
)


def confirmation_request(record: ConversationRecord) -> list[ChatMessage]:
    skipped = ", ".join(record.skipped_fields) or "none"
    return [
        ChatMessage(
            "user",
            f"Student profile: {_known_values(record.profile)}\nSkipped: {skipped}\n\n"
            "Summarize this profile for the student in one or two short sentences and ask whether "
            f"it is correct. Write in language code '{record.language}'.",
        )
    ]


def recommendation_request(
    record: ConversationRecord, eligible: list[CatalogEntry], limit: int
) -> list[ChatMessage]:
    options = [
        entry.model_dump(mode="json", exclude={"min_school_year", "min_free_hours_required"})
        for entry in eligible
    ]
    return [
        ChatMessage(
            "user",
            f"Student profile: {_known_values(record.profile)}\n\n"
            f"Eligible clubs and classes:\n{json.dumps(options, ensure_ascii=False)}\n\n"
            f"Choose up to {limit} that best fit the student's magic orientation, learning "
            "modality and interests. Use the exact names from the list. Give each a one-sentence "
            "reason. Then write `reply`: a short spoken message presenting them and saying "
            f"goodbye. Write reasons and reply in language code '{record.language}'.",
        )
    ]


def no_match_request(
    record: ConversationRecord, rejected: dict[str, list[str]]
) -> list[ChatMessage]:
    return [
        ChatMessage(
            "user",
            f"Student profile: {_known_values(record.profile)}\n\n"
            f"No club or class is open to this student yet. Reasons per option:\n"
            f"{json.dumps(rejected, ensure_ascii=False)}\n\n"
            "In two or three short sentences, kindly explain the main reasons and what would open "
            f"options up (e.g. more free hours or a later year). Write in language code "
            f"'{record.language}'.",
        )
    ]


def summary_request(record: ConversationRecord) -> list[ChatMessage]:
    transcript = "\n".join(f"{m.role}: {m.text}" for m in record.transcript)
    recs = json.dumps([r.model_dump() for r in record.recommendations], ensure_ascii=False)
    sentiments = ", ".join(s.value for s in record.sentiment_history) or "none"
    return [
        ChatMessage(
            "user",
            f"Transcript:\n{transcript}\n\nFinal profile: {_known_values(record.profile)}\n"
            f"Skipped fields: {', '.join(record.skipped_fields) or 'none'}\n"
            f"Recommendations: {recs}\n"
            f"Student sentiment per message: {sentiments}\n\n"
            "Write `key_points` in English for school staff: 3-6 short sentences covering the "
            "student's profile, anything notable they said or how they felt (concerns, "
            "frustration, corrections, values worth double-checking) and the recommendations.",
        )
    ]


def fallback_confirmation(record: ConversationRecord) -> str:
    """English confirmation used when the LLM call fails."""
    values = ", ".join(
        f"{name.replace('_', ' ')}: {value}"
        for name, value in record.profile.model_dump(mode="json", exclude_none=True).items()
    )
    return f"Here's what I have: {values}. Is that correct?"


def fallback_recommendations(recommendations: list[Recommendation]) -> str:
    """English recommendation message used when the LLM call fails or invents names."""
    if not recommendations:
        return "I couldn't find a club or class that fits you right now. Thanks for your time!"
    items = "; ".join(f"{r.name}: {r.reason}" for r in recommendations)
    return f"Here's what I recommend: {items}. Good luck this term!"
