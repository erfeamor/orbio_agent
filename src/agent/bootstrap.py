"""Builds the real components from settings. Interfaces (CLI, API) only call these."""

from pydantic import SecretStr

from agent.config import Settings
from agent.domain.catalog import load_catalog
from agent.engine.engine import Engine
from agent.llm.claude import ClaudeLLM
from agent.speech.base import STTClient, TTSClient
from agent.speech.deepgram import DeepgramSTT
from agent.speech.elevenlabs import ElevenLabsSTT, ElevenLabsTTS
from agent.storage.file_repo import JsonFileRepository


def _secret(value: SecretStr | None) -> str | None:
    # An empty KEY= line in .env means "not set".
    return (value.get_secret_value() if value else "") or None


def build_engine(settings: Settings) -> Engine:
    llm = ClaudeLLM(
        settings.anthropic_model,
        effort=settings.llm_effort,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        api_key=_secret(settings.anthropic_api_key),
    )
    return Engine(
        llm, JsonFileRepository(settings.conversations_dir), load_catalog(settings.catalog_path)
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
