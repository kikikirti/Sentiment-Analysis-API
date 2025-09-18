from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

# Ensure model exists before importing app routes (one-time safety)
if not Path(settings.MODEL_PATH).exists():
    from model.train import train_and_save

    train_and_save(settings.MODEL_PATH)

client = TestClient(app)


def test_predict_requires_api_key():
    r = client.post("/predict", json={"text": "i love this"})
    assert r.status_code == 401


def test_predict_positive():
    r = client.post(
        "/predict",
        headers={"X-API-Key": settings.API_KEY},
        json={"text": "absolutely fantastic quality, highly recommend"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in {"positive", "negative"}
    assert 0.0 <= data["score"] <= 1.0
    assert "explain" in data


def test_predict_batch_shape():
    r = client.post(
        "/predict/batch",
        headers={"X-API-Key": settings.API_KEY},
        json={"texts": ["i love this", "terrible product"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 2
    for item in data:
        assert item["label"] in {"positive", "negative"}
        assert 0.0 <= item["score"] <= 1.0
