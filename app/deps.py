# app/deps.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from app.config import settings

logger = logging.getLogger("api")

# Cached artifact (loaded once per process)
_ARTIFACT: dict[str, Any] | None = None

def _load_artifact() -> tuple[Any, dict[str, Any]]:
    global _ARTIFACT
    if _ARTIFACT is None:
        p = Path(settings.MODEL_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found at {p}. Run: python model/train.py")
        _ARTIFACT = joblib.load(p)
        meta = _ARTIFACT.get("meta", {}) if isinstance(_ARTIFACT, dict) else {}
        logger.info(
            "Model loaded from %s (name=%s, version=%s)",
            p,
            meta.get("model_name", "unknown"),
            meta.get("version", "unknown"),
        )
    if isinstance(_ARTIFACT, dict) and "pipeline" in _ARTIFACT:
        return _ARTIFACT["pipeline"], _ARTIFACT.get("meta", {})
    # Back-compat: artifact directly is a pipeline
    return _ARTIFACT, {}  # type: ignore[return-value]

def get_pipeline_and_meta():
    return _load_artifact()
