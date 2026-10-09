"""HTTP API used by the voice frontend. Run with ``make serve``."""

import base64
import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Path, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.bootstrap import build_engine, build_stt, build_tts
from agent.config import Settings
from agent.domain.models import (
    ConversationRecord,
    Phase,
    Recommendation,
    StudentProfile,
)
from agent.engine.engine import ConversationNotFoundError, Engine
from agent.llm.base import LLMConfigurationError
from agent.speech.base import STTClient, TTSClient
from agent.speech.pipeline import VoicePipeline, speak_or_none
from agent.storage.base import VersionConflictError

logger = logging.getLogger(__name__)

ConversationId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]


@dataclass(frozen=True)
class Components:
    engine: Engine
    stt: STTClient | None
    tts: TTSClient | None


class StartRequest(BaseModel):
    speak: bool = Field(default=False, description="Include the reply as audio")


class MessageRequest(BaseModel):
    text: str = Field(max_length=10_000)
    speak: bool = False


class TurnResponse(BaseModel):
    conversation_id: str
    transcript: str | None = Field(default=None, description="What STT heard (audio turns)")
    reply: str
    audio_base64: str | None = None
    audio_media_type: str | None = None
    phase: Phase
    language: str
    profile: StudentProfile
    field_errors: dict[str, str]
    recommendations: list[Recommendation]
    summary: list[str] | None


def create_app(components: Components | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if components is None:
        components = Components(
            engine=build_engine(settings), stt=build_stt(settings), tts=build_tts(settings)
        )
    engine, tts = components.engine, components.tts
    voice = (
        VoicePipeline(engine, components.stt, tts, min_confidence=settings.stt_min_confidence)
        if components.stt
        else None
    )

    app = FastAPI(title="Arcanum Academy advisor")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def respond(
        record: ConversationRecord,
        reply: str,
        audio: bytes | None = None,
        transcript: str | None = None,
    ) -> TurnResponse:
        return TurnResponse(
            conversation_id=record.id,
            transcript=transcript,
            reply=reply,
            audio_base64=base64.b64encode(audio).decode() if audio else None,
            audio_media_type=tts.media_type if audio and tts else None,
            phase=record.phase,
            language=record.language,
            profile=record.profile,
            field_errors=record.field_errors,
            recommendations=record.recommendations,
            summary=record.summary,
        )

    def speak(text: str, wanted: bool) -> bytes | None:
        return speak_or_none(tts, text) if wanted else None

    @app.exception_handler(ConversationNotFoundError)
    def not_found(_: Request, exc: ConversationNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "conversation not found"})

    @app.exception_handler(VersionConflictError)
    def conflict(_: Request, exc: VersionConflictError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": "the conversation changed while this message was processed"},
        )

    @app.exception_handler(LLMConfigurationError)
    def llm_misconfigured(_: Request, exc: LLMConfigurationError) -> JSONResponse:
        logger.error("LLM configuration error: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "the advisor is not configured"})

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "speech_to_text": voice is not None,
            "text_to_speech": tts is not None,
        }

    @app.post("/sessions", status_code=201)
    def start_session(body: StartRequest | None = None) -> TurnResponse:
        result = engine.start()
        return respond(result.record, result.reply, speak(result.reply, bool(body and body.speak)))

    @app.get("/sessions/{conversation_id}")
    def get_session(conversation_id: ConversationId) -> ConversationRecord:
        record = engine.get(conversation_id)
        if record is None:
            raise ConversationNotFoundError(conversation_id)
        return record

    @app.post("/sessions/{conversation_id}/messages")
    def send_message(conversation_id: ConversationId, body: MessageRequest) -> TurnResponse:
        result = engine.handle_turn(conversation_id, body.text)
        return respond(result.record, result.reply, speak(result.reply, body.speak))

    @app.post("/sessions/{conversation_id}/audio")
    def send_audio(
        conversation_id: ConversationId, audio: Annotated[UploadFile, File()]
    ) -> TurnResponse:
        if voice is None:
            raise HTTPException(503, "speech-to-text is not configured")
        content_type = (audio.content_type or "").split(";")[0].strip()
        if not content_type.startswith(("audio/", "video/webm")):
            raise HTTPException(415, f"expected an audio file, got {content_type or 'unknown'}")
        data = audio.file.read(settings.max_audio_bytes + 1)
        if len(data) > settings.max_audio_bytes:
            raise HTTPException(413, "audio clip is too large")
        if not data:
            raise HTTPException(400, "audio clip is empty")

        turn = voice.handle(conversation_id, data, content_type)
        return respond(turn.record, turn.reply, turn.audio, turn.transcript)

    return app
