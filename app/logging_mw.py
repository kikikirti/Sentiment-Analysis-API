# app/logging_mw.py
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime

from fastapi import Request
from starlette.responses import Response

logger = logging.getLogger("api")


def _utc_now_iso() -> str:
    # RFC3339-ish Zulu time
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = req_id  # available to handlers if needed

    response: Response | None = None
    error: Exception | None = None
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        error = exc
        dur_ms = (time.perf_counter() - start) * 1000
        log = {
            "ts": _utc_now_iso(),
            "event": "request",
            "request_id": req_id,
            "method": request.method,
            "path": request.url.path,
            "query": request.url.query or None,
            "status": 500,
            "duration_ms": round(dur_ms, 2),
            "client_ip": getattr(request.client, "host", None),
            "user_agent": request.headers.get("user-agent"),
            "error": repr(exc),
        }
        # Log the structured line with stacktrace
        logger.exception(json.dumps(log))
        raise
    finally:
        if error is None:
            dur_ms = (time.perf_counter() - start) * 1000
            status = getattr(response, "status_code", None)
            log = {
                "ts": _utc_now_iso(),
                "event": "request",
                "request_id": req_id,
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query or None,
                "status": status,
                "duration_ms": round(dur_ms, 2),
                "client_ip": getattr(request.client, "host", None),
                "user_agent": request.headers.get("user-agent"),
            }
            # Attach request id to the response for tracing
            if response is not None:
                response.headers["X-Request-ID"] = req_id
            logger.info(json.dumps(log))
