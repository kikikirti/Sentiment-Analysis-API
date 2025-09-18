# app/logging_mw.py
from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.responses import Response

logger = logging.getLogger("api")


async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
        return response
    finally:
        dur_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.2f ms)",
            request.method,
            request.url.path,
            getattr(response, "status_code", "?"),
            dur_ms,
        )
