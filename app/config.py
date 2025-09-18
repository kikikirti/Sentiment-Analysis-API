# app/config.py
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_KEY: str = "your_key_here"
    LOG_LEVEL: str = "INFO"
    MODEL_PATH: str = "model/pipeline.pkl"
    MAX_TEXT_LEN: int = 10_000
    MAX_BATCH: int = 200
    MAX_BODY_BYTES: int = 10_240  # ~10KB request payload cap  ← add this

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
