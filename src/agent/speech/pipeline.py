"""Voice turn: audio in, transcribe, run the engine, speak the reply."""

import logging
import time
from dataclasses import dataclass

from agent.domain.models import ConversationRecord
from agent.engine import prompts
from agent.engine.engine import ConversationNotFoundError, Engine
from agent.speech.base import SpeechError, STTClient, TTSClient

logger = logging.getLogger(__name__)

HEARING_FAILURE_MESSAGE = "Sorry, I couldn't hear that properly. Could you try again?"


def speak_or_none(tts: TTSClient | None, text: str) -> bytes | None:
    """Synthesize ``text``; on failure the caller still has the text reply."""
    if tts is None:
        return None
    try:
        return tts.synthesize(text)
    except SpeechError:
        logger.exception("speech synthesis failed; replying with text only")
        return None


@dataclass(frozen=True)
class VoiceTurn:
    transcript: str | None
    reply: str
    audio: bytes | None
    record: ConversationRecord


class VoicePipeline:
    def __init__(
        self,
        engine: Engine,
        stt: STTClient,
        tts: TTSClient | None,
        *,
        min_confidence: float = 0.4,
    ) -> None:
        self._engine = engine
        self._stt = stt
        self._tts = tts
        self._min_confidence = min_confidence

    def handle(self, conversation_id: str, audio: bytes, content_type: str) -> VoiceTurn:
        # Fail before paying for a transcription if the conversation doesn't exist.
        if self._engine.get(conversation_id) is None:
            raise ConversationNotFoundError(conversation_id)
        started = time.monotonic()
        try:
            transcript = self._stt.transcribe(audio, content_type)
        except SpeechError:
            logger.exception("transcription failed for conversation %s", conversation_id)
            return self._without_turn(conversation_id, None, HEARING_FAILURE_MESSAGE)
        stt_ms = (time.monotonic() - started) * 1000

        if not transcript.text or transcript.confidence < self._min_confidence:
            logger.info(
                "unclear audio (confidence %.2f, %d chars)",
                transcript.confidence,
                len(transcript.text),
            )
            return self._without_turn(
                conversation_id, transcript.text or None, prompts.EMPTY_INPUT_MESSAGE
            )

        result = self._engine.handle_turn(conversation_id, transcript.text)
        engine_ms = (time.monotonic() - started) * 1000 - stt_ms
        speech = speak_or_none(self._tts, result.reply)
        logger.info(
            "voice turn stt_ms=%d engine_ms=%d tts_ms=%d",
            stt_ms,
            engine_ms,
            (time.monotonic() - started) * 1000 - stt_ms - engine_ms,
        )
        return VoiceTurn(transcript.text, result.reply, speech, result.record)

    def _without_turn(self, conversation_id: str, transcript: str | None, reply: str) -> VoiceTurn:
        """Reply without running the engine; the stored conversation is unchanged."""
        record = self._engine.get(conversation_id)
        if record is None:
            raise ConversationNotFoundError(conversation_id)
        return VoiceTurn(transcript, reply, speak_or_none(self._tts, reply), record)
