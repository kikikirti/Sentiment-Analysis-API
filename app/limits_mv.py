# app/limits_mw.py
from __future__ import annotations

import logging

from fastapi import Request
from starlette.responses import JSONResponse, Response

from app.config import settings

logger = logging.getLogger("api")

async def body_limit_middleware(request: Request, call_next):
    """
    Lightweight payload size guard.
    - If Content-Length is present and > MAX_BODY_BYTES -> 413 fast reject.
    - If header missing, we skip to avoid consuming the body; field-level
      validators (MAX_TEXT_LEN / MAX_BATCH) still protect us.
    """
    cl = request.headers.get("content-length")
    try:
        if cl is not None and int(cl) > settings.MAX_BODY_BYTES:
            return JSONResponse(
                {"detail": f"Payload too large (>{settings.MAX_BODY_BYTES} bytes)"},
                status_code=413,
            )
    except ValueError:
        # Malformed header -> let request proceed; downstream will 400/422.
        pass

    resp: Response = await call_next(request)
    return resp
