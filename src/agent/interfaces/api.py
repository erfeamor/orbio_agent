"""HTTP API used by the voice frontend. Run with ``make serve``."""

import base64
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Path, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from agent.bootstrap import build_engine, build_stt, build_tts
from agent.config import Settings
from agent.domain.models import (
    ConversationRecord,
    EndReason,
    Message,
    Phase,
    ProfileField,
    Recommendation,
    Sentiment,
    StudentProfile,
)
from agent.engine.engine import ConversationNotFoundError, Engine
from agent.llm.base import LLMConfigurationError
from agent.logging_config import bound, configure_logging, request_id_var
from agent.speech.base import STTClient, TTSClient
from agent.speech.pipeline import VoicePipeline, speak_or_none
from agent.storage.base import (
    CONVERSATION_ID_PATTERN,
    StorageUnavailableError,
    VersionConflictError,
)

logger = logging.getLogger(__name__)

ConversationId = Annotated[str, Path(pattern=CONVERSATION_ID_PATTERN.pattern)]

REQUEST_ID_HEADER = "X-Request-ID"
# Client-supplied ids go into logs, so only accept a short, safe format.
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Hard cap on a text request; the engine answers anything above MAX_INPUT_CHARS politely.
MAX_REQUEST_CHARS = 10_000
# Browsers record "audio/webm", "audio/ogg" or "audio/mp4"; some label WebM as "video/webm".
ACCEPTED_AUDIO_TYPES = ("audio/", "video/webm")


@dataclass(frozen=True)
class Components:
    engine: Engine
    stt: STTClient | None
    tts: TTSClient | None


class StartRequest(BaseModel):
    speak: bool = Field(default=False, description="Include the reply as audio")
    language: str | None = Field(
        default=None,
        max_length=35,
        description="Preferred language (e.g. the browser's 'es-ES'); picks the consent text",
    )


class MessageRequest(BaseModel):
    text: str = Field(max_length=MAX_REQUEST_CHARS)
    speak: bool = False


class SessionView(BaseModel):
    """A conversation as the API shows it. Storage details (version, model, prompt version)
    stay internal, so the stored record can change without changing the API."""

    conversation_id: str
    phase: Phase
    end_reason: EndReason | None
    language: str
    created_at: datetime
    updated_at: datetime
    transcript: list[Message]
    profile: StudentProfile
    skipped_fields: list[ProfileField]
    field_errors: dict[str, str]
    sentiment_history: list[Sentiment]
    recommendations: list[Recommendation]
    summary: list[str] | None

    @classmethod
    def of(cls, record: ConversationRecord) -> "SessionView":
        return cls(
            conversation_id=record.id,
            **record.model_dump(
                include={
                    "phase",
                    "end_reason",
                    "language",
                    "created_at",
                    "updated_at",
                    "transcript",
                    "profile",
                    "skipped_fields",
                    "field_errors",
                    "sentiment_history",
                    "recommendations",
                    "summary",
                }
            ),
        )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    speech_to_text: bool
    stt_provider: str | None = Field(description="Speech-to-text provider name, if configured")
    text_to_speech: bool
    max_input_chars: int = Field(description="Longer messages are answered with a polite error")


class TurnResponse(BaseModel):
    conversation_id: str
    transcript: str | None = Field(default=None, description="What STT heard (audio turns)")
    reply: str
    audio_base64: str | None = None
    audio_media_type: str | None = None
    phase: Phase
    end_reason: EndReason | None
    language: str
    profile: StudentProfile
    field_errors: dict[str, str]
    recommendations: list[Recommendation]
    summary: list[str] | None


def create_app(components: Components | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
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
        allow_headers=["Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
    )

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Tag every log line of a request with an id, returned to the client for support."""
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        rid = incoming if _REQUEST_ID_PATTERN.fullmatch(incoming) else uuid4().hex
        with bound(request_id_var, rid):
            response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response

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
            end_reason=record.end_reason,
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
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"detail": "conversation not found"}
        )

    @app.exception_handler(VersionConflictError)
    def conflict(_: Request, exc: VersionConflictError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "the conversation changed while this message was processed"},
        )

    @app.exception_handler(StorageUnavailableError)
    def storage_unavailable(_: Request, exc: StorageUnavailableError) -> JSONResponse:
        logger.error("storage unavailable: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "storage is temporarily unavailable, try again"},
        )

    @app.exception_handler(LLMConfigurationError)
    def llm_misconfigured(_: Request, exc: LLMConfigurationError) -> JSONResponse:
        logger.error("LLM configuration error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "the advisor is not configured"},
        )

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(
            speech_to_text=voice is not None,
            stt_provider=components.stt.name if components.stt else None,
            text_to_speech=tts is not None,
            max_input_chars=settings.max_input_chars,
        )

    @app.post("/sessions", status_code=status.HTTP_201_CREATED)
    def start_session(body: StartRequest | None = None) -> TurnResponse:
        result = engine.start(body.language if body else None)
        return respond(result.record, result.reply, speak(result.reply, bool(body and body.speak)))

    @app.get("/sessions/{conversation_id}")
    def get_session(conversation_id: ConversationId) -> SessionView:
        record = engine.get(conversation_id)
        if record is None:
            raise ConversationNotFoundError(conversation_id)
        return SessionView.of(record)

    @app.post("/sessions/{conversation_id}/messages")
    def send_message(conversation_id: ConversationId, body: MessageRequest) -> TurnResponse:
        result = engine.handle_turn(conversation_id, body.text)
        return respond(result.record, result.reply, speak(result.reply, body.speak))

    @app.post("/sessions/{conversation_id}/audio")
    def send_audio(
        conversation_id: ConversationId, audio: Annotated[UploadFile, File()]
    ) -> TurnResponse:
        if voice is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "speech-to-text is not configured"
            )
        content_type = (audio.content_type or "").split(";")[0].strip()
        if not content_type.startswith(ACCEPTED_AUDIO_TYPES):
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"expected an audio file, got {content_type or 'unknown'}",
            )
        data = audio.file.read(settings.max_audio_bytes + 1)
        if len(data) > settings.max_audio_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "audio clip is too large")
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "audio clip is empty")

        turn = voice.handle(conversation_id, data, content_type)
        return respond(turn.record, turn.reply, turn.audio, turn.transcript)

    return app


def app_factory() -> FastAPI:
    """Entry point for uvicorn (``make serve``): reads settings and sets up logging once."""
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    return create_app(settings=settings)
