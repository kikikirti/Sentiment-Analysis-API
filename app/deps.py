# app/deps.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from app.config import settings

_ARTIFACT: dict[str, Any] | None = None


def _load_artifact() -> tuple[Any, dict[str, Any]]:
    global _ARTIFACT
    if _ARTIFACT is None:
        p = Path(settings.MODEL_PATH)
        if not p.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {p}. Run: python model/train.py"
            )
        _ARTIFACT = joblib.load(p)
    if isinstance(_ARTIFACT, dict) and "pipeline" in _ARTIFACT:
        return _ARTIFACT["pipeline"], _ARTIFACT.get("meta", {})
    # Back-compat: raw pipeline (no meta)
    return _ARTIFACT, {}  # type: ignore[return-value]


def get_pipeline_and_meta():
    return _load_artifact()
