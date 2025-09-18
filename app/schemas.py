# app/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class PredictOut(BaseModel):
    label: str
    score: float
    explain: dict | None = None


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=200)


class BatchOutItem(BaseModel):
    label: str
    score: float
