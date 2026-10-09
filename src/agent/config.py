from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    storage: Literal["file", "s3"] = "file"
    conversations_dir: Path = Path("./conversations")
    catalog_path: Path = Path("./data/catalog.json")

    # Read here so a key in .env works; if unset, the SDK falls back to its own credential lookup.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 2

    # Speech is optional: without keys, the text endpoints still work and replies have no audio.
    stt_provider: Literal["deepgram", "elevenlabs"] = "deepgram"
    deepgram_api_key: SecretStr | None = None
    deepgram_model: str = "nova-3"
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    elevenlabs_model: str = "eleven_flash_v2_5"
    elevenlabs_stt_model: str = "scribe_v2"
    speech_timeout_seconds: float = 15.0
    stt_min_confidence: float = 0.4

    # HTTP API
    cors_origins: list[str] = ["http://localhost:5173"]
    max_audio_bytes: int = 5_000_000
