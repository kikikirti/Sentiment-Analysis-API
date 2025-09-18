from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.MAX_TEXT_LEN)

    @field_validator("text")
    @classmethod
    def not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v


class PredictOut(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    # keep this so existing code/tests don't break
    explain: dict | None = None


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=settings.MAX_BATCH)


class BatchOutItem(BaseModel):
    label: str
    score: float
