"""Fixed messages for the student, used when no LLM reply is available or wanted.

The LLM replies in any language; these cover the ones listed in MESSAGES and fall back to
English. Adding a language means adding one Messages entry.
"""

from dataclasses import dataclass

from agent.domain.catalog import RankedEntry
from agent.domain.models import DEFAULT_LANGUAGE, Recommendation, StudentProfile


@dataclass(frozen=True)
class Messages:
    consent: str
    empty_input: str
    too_long: str
    llm_failure: str
    refusal: str
    finished: str
    hearing_failure: str
    # Fallbacks for LLM calls that failed (or invented catalog names).
    confirmation: str  # {values}
    recommendations: str  # {items}
    no_match: str
    field_labels: dict[str, str]
    orientation_match: str  # {value}
    modality_match: str  # {value}
    reason_separator: str
    no_reason: str

    def fallback_confirmation(self, profile: StudentProfile) -> str:
        values = ", ".join(
            f"{self.field_labels.get(name, name)}: {value}"
            for name, value in profile.model_dump(mode="json", exclude_none=True).items()
        )
        return self.confirmation.format(values=values)

    def fallback_recommendations(self, recommendations: list[Recommendation]) -> str:
        if not recommendations:
            return self.no_match
        items = "; ".join(f"{r.name}: {r.reason}" for r in recommendations)
        return self.recommendations.format(items=items)

    def match_reason(self, ranked: RankedEntry) -> str:
        """Why a deterministically ranked entry fits the student."""
        ideal = ranked.entry.ideal_profile
        parts = []
        if ranked.matches_orientation:
            parts.append(self.orientation_match.format(value=ideal.magic_orientation.value))
        if ranked.matches_modality:
            parts.append(self.modality_match.format(value=ideal.learning_modality.value))
        return self.reason_separator.join(parts) or self.no_reason


_ENGLISH = Messages(
    consent=(
        "Welcome to the Arcanum Academy advisor! I'll ask a few quick questions about your "
        "school year, specialization, interests, what drives your magic, how you like to learn "
        "and your free time, so I can recommend clubs and advanced classes. Your answers are "
        "saved so the school can follow up. Shall we begin?"
    ),
    empty_input="Sorry, I didn't catch that. Could you say it again?",
    too_long="That's a bit long for me. Could you answer in a shorter message?",
    llm_failure="Sorry, I had trouble thinking just now. Could you repeat that?",
    refusal="I can't help with that. Let's get back to finding you the right clubs and classes.",
    finished="This conversation has ended. Start a new one any time!",
    hearing_failure="Sorry, I couldn't hear that properly. Could you try again?",
    confirmation="Here's what I have: {values}. Is that correct?",
    recommendations="Here's what I recommend: {items}. Good luck this term!",
    no_match="I couldn't find a club or class that fits you right now. Thanks for your time!",
    field_labels={
        "school_year": "school year",
        "specialization": "specialization",
        "interests": "interests",
        "magic_orientation": "magic orientation",
        "learning_modality": "learning style",
        "weekly_free_hours": "free hours per week",
    },
    orientation_match="suits your {value} drive",
    modality_match="fits your {value} learning style",
    reason_separator=" and ",
    no_reason="open to you",
)

_SPANISH = Messages(
    consent=(
        "¡Te damos la bienvenida al orientador de la Academia Arcanum! Te haré unas preguntas "
        "rápidas sobre tu curso, tu especialización, tus intereses, qué impulsa tu magia, cómo "
        "aprendes mejor y tu tiempo libre, para recomendarte clubes y clases avanzadas. Tus "
        "respuestas se guardan para que la escuela pueda hacer seguimiento. ¿Empezamos?"
    ),
    empty_input="Perdona, no te he entendido. ¿Puedes repetirlo?",
    too_long="Es un poco largo para mí. ¿Puedes responder con un mensaje más corto?",
    llm_failure="Perdona, he tenido un problema al pensar. ¿Puedes repetirlo?",
    refusal="No puedo ayudarte con eso. Volvamos a buscar los clubes y clases adecuados para ti.",
    finished="Esta conversación ha terminado. ¡Puedes empezar otra cuando quieras!",
    hearing_failure="Perdona, no te he oído bien. ¿Puedes intentarlo de nuevo?",
    confirmation="Esto es lo que tengo: {values}. ¿Es correcto?",
    recommendations="Te recomiendo: {items}. ¡Mucha suerte este curso!",
    no_match="Ahora mismo no encuentro un club o clase para ti. ¡Gracias por tu tiempo!",
    field_labels={
        "school_year": "curso",
        "specialization": "especialización",
        "interests": "intereses",
        "magic_orientation": "orientación mágica",
        "learning_modality": "forma de aprender",
        "weekly_free_hours": "horas libres por semana",
    },
    orientation_match="encaja con tu impulso {value}",
    modality_match="encaja con tu forma de aprender {value}",
    reason_separator=" y ",
    no_reason="disponible para ti",
)

MESSAGES: dict[str, Messages] = {"en": _ENGLISH, "es": _SPANISH}


def messages_for(language: str | None) -> Messages:
    return MESSAGES.get(language or DEFAULT_LANGUAGE, MESSAGES[DEFAULT_LANGUAGE])
