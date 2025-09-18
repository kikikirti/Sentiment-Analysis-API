from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(scope="session")
def api_key_header() -> dict[str, str]:
    return {"X-API-Key": settings.API_KEY}


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
