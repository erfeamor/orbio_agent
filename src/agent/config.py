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
