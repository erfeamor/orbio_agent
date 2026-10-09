"""Builds the real components from settings. Interfaces (CLI, API) only call these."""

from typing import assert_never

from pydantic import SecretStr

from agent.config import Settings
from agent.domain.catalog import load_catalog
from agent.engine.engine import Engine, EngineConfig
from agent.engine.recommender import Recommender
from agent.engine.summarizer import Summarizer
from agent.llm.base import LLMClient
from agent.llm.claude import ClaudeLLM
from agent.speech.base import STTClient, TTSClient
from agent.speech.deepgram import DeepgramSTT
from agent.speech.elevenlabs import ElevenLabsSTT, ElevenLabsTTS
from agent.storage.base import ConversationRepository
from agent.storage.file_repo import JsonFileRepository


def _secret(value: SecretStr | None) -> str | None:
    # An empty KEY= line in .env means "not set".
    return (value.get_secret_value() if value else "") or None


# Extensible if other LLM providers are needed; the engine is built with a single LLM.
def build_llm(settings: Settings) -> LLMClient:
    return ClaudeLLM(
        settings.anthropic_model,
        effort=settings.llm_effort,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        api_key=_secret(settings.anthropic_api_key),
    )


def build_repository(settings: Settings) -> ConversationRepository:
    match settings.storage:
        case "file":
            return JsonFileRepository(settings.conversations_dir)
        case "s3":
            return _build_s3_repository(settings)
        case unreachable:
            # mypy fails if a new value is added to the Literal but not handled here.
            assert_never(unreachable)


def _build_s3_repository(settings: Settings) -> ConversationRepository:
    # Imported here so local installs don't need boto3 (the `aws` extra).
    import boto3
    from botocore.config import Config

    from agent.storage.s3_repo import S3ConversationRepository

    if not settings.s3_bucket:  # also checked when Settings loads
        raise ValueError("STORAGE=s3 requires S3_BUCKET")
    client = boto3.client(
        "s3",
        config=Config(
            connect_timeout=settings.s3_timeout_seconds,
            read_timeout=settings.s3_timeout_seconds,
            retries={"max_attempts": settings.s3_max_attempts, "mode": "standard"},
        ),
    )
    return S3ConversationRepository(
        client=client, bucket=settings.s3_bucket, prefix=settings.s3_prefix
    )


def build_engine(settings: Settings) -> Engine:
    llm = build_llm(settings)
    catalog = load_catalog(settings.catalog_path)
    return Engine(
        llm,
        build_repository(settings),
        Recommender(llm, catalog, limit=settings.max_recommendations),
        Summarizer(llm),
        EngineConfig(
            max_input_chars=settings.max_input_chars,
            frustration_streak=settings.frustration_streak,
        ),
    )


def build_stt(settings: Settings) -> STTClient | None:
    """The provider is chosen with STT_PROVIDER; ElevenLabs reuses the TTS key."""
    if settings.stt_provider == "elevenlabs":
        api_key = _secret(settings.elevenlabs_api_key)
        if api_key is None:
            return None
        return ElevenLabsSTT(
            api_key, model=settings.elevenlabs_stt_model, timeout=settings.speech_timeout_seconds
        )
    api_key = _secret(settings.deepgram_api_key)
    if api_key is None:
        return None
    return DeepgramSTT(
        api_key, model=settings.deepgram_model, timeout=settings.speech_timeout_seconds
    )


def build_tts(settings: Settings) -> TTSClient | None:
    api_key = _secret(settings.elevenlabs_api_key)
    if api_key is None:
        return None
    return ElevenLabsTTS(
        api_key,
        voice_id=settings.elevenlabs_voice_id,
        model=settings.elevenlabs_model,
        timeout=settings.speech_timeout_seconds,
    )
