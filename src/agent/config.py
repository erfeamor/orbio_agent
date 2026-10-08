from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    storage: Literal["file", "s3"] = "file"
    conversations_dir: Path = Path("./conversations")
    catalog_path: Path = Path("./data/catalog.json")
