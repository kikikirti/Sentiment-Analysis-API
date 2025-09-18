# app/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env if present


@dataclass
class Settings:
    API_KEY: str = os.getenv("API_KEY", "your_key_here")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model/pipeline.pkl")
    MAX_TEXT_LEN: int = int(os.getenv("MAX_TEXT_LEN", "10000"))
    MAX_BATCH: int = int(os.getenv("MAX_BATCH", "200"))


settings = Settings()

# Ensure model dir exists (nice for local dev)
Path(settings.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
