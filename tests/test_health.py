from __future__ import annotations


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_meta_shape(client):
    r = client.get("/meta")
    assert r.status_code == 200
    data = r.json()
    for key in ("model_name", "version", "trained_on", "labels"):
        assert key in data
    assert isinstance(data["labels"], list) and len(data["labels"]) >= 2
