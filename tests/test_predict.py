from __future__ import annotations

from app.config import settings


def test_predict_requires_api_key(client):
    r = client.post("/predict", json={"text": "ok"})
    assert r.status_code == 401


def test_predict_happy_path(client, api_key_header):
    # Use a string that is stable with the tiny model; this repo tends to label it "negative".
    r = client.post("/predict", headers=api_key_header, json={"text": "terrible product"})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) >= {"label", "score"}
    assert 0.0 <= float(body["score"]) <= 1.0
    # Deterministic: same input → same output
    r2 = client.post("/predict", headers=api_key_header, json={"text": "terrible product"})
    body2 = r2.json()
    assert body2["label"] == body["label"]
    assert body2["score"] == body["score"]  # rounded in the API to 6 dp


def test_predict_validation_empty(client, api_key_header):
    # Missing field -> 422
    r = client.post("/predict", headers=api_key_header, json={})
    assert r.status_code == 422
    # Empty string -> 422 (pydantic field validator)
    r2 = client.post("/predict", headers=api_key_header, json={"text": ""})
    assert r2.status_code == 422


def test_predict_batch_ok(client, api_key_header):
    payload = {"texts": ["i love this", "pleasant surprise", "terrible product"]}
    r = client.post("/predict/batch", headers=api_key_header, json=payload)
    assert r.status_code == 200
    out = r.json()
    assert isinstance(out, list) and len(out) == 3
    for item in out:
        assert set(item.keys()) >= {"label", "score"}
        assert 0.0 <= float(item["score"]) <= 1.0


def test_predict_batch_cap(client, api_key_header):
    # Build a payload larger than the allowed MAX_BATCH
    too_many = [f"sample {i}" for i in range(settings.MAX_BATCH + 1)]
    r = client.post("/predict/batch", headers=api_key_header, json={"texts": too_many})
    # Depending on whether schema or route guard triggers first, we accept either 422 or 413.
    assert r.status_code in (422, 413), f"unexpected status: {r.status_code}, body={r.text}"
