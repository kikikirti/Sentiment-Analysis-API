# tests/test_health.py
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

