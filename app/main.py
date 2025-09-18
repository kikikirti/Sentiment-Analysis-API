# app/main.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from sklearn.pipeline import Pipeline

from app.config import settings
from app.deps import get_pipeline_and_meta
from app.logging_mw import timing_middleware
from app.schemas import BatchIn, BatchOutItem, PredictIn, PredictOut

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))

# ---------- auth ----------
def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

# ---------- helpers ----------
def _predict_text(pipe: Pipeline, text: str) -> tuple[str, float, list[str]]:
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba([text])[0]
        classes = pipe.classes_ if hasattr(pipe, "classes_") else pipe.named_steps["clf"].classes_
        idx = int(np.argmax(probs))
        score = float(probs[idx])
        label = str(classes[idx])
    else:
        df = pipe.decision_function([text])[0]
        score = float(1 / (1 + np.exp(-abs(df))))
        label = str(pipe.predict([text])[0])

    top_tokens: list[str] = []
    if hasattr(pipe, "named_steps") and "tfidf" in pipe.named_steps:
        vec = pipe.named_steps["tfidf"]
        X = vec.transform([text])
        if hasattr(X, "tocoo"):
            X = X.tocoo()
            import numpy as _np
            top_idx = _np.argsort(X.data)[-3:][::-1]
            feats = _np.array(vec.get_feature_names_out())
            top_tokens = [feats[i] for i in X.col[top_idx]]
    return label, score, top_tokens

# ---------- router (no global `app` needed) ----------
router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/meta")
def meta():
    pipe, meta_dict = get_pipeline_and_meta()
    labels = meta_dict.get("labels")
    if labels is None and hasattr(pipe, "named_steps"):
        clf = pipe.named_steps.get("clf")
        if hasattr(clf, "classes_"):
            labels = clf.classes_.tolist()
    return {
        "model_name": meta_dict.get("model_name", "sklearn + tfidf"),
        "version": meta_dict.get("version", "0.0.0"),
        "trained_on": meta_dict.get("trained_on", "unknown"),
        "labels": labels or ["negative", "positive"],
    }

@router.post("/predict", response_model=PredictOut, dependencies=[Depends(require_api_key)])
def predict(payload: PredictIn):
    pipe, _ = get_pipeline_and_meta()
    label, score, top_tokens = _predict_text(pipe, payload.text)
    return {"label": label, "score": round(score, 6), "explain": {"top_tokens": top_tokens}}

@router.post(
    "/predict/batch",
    response_model=list[BatchOutItem],
    dependencies=[Depends(require_api_key)],
)
def predict_batch(payload: BatchIn):
    pipe, _ = get_pipeline_and_meta()
    out: list[BatchOutItem] = []
    for t in payload.texts:
        label, score, _ = _predict_text(pipe, t)
        out.append({"label": label, "score": round(score, 6)})
    return out

# ---------- lifespan & app factory ----------
@asynccontextmanager
async def lifespan(app_: FastAPI):
    # warm the model once at startup
    get_pipeline_and_meta()
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)
    app.middleware("http")(timing_middleware)
    app.include_router(router)
    return app

app = create_app()
