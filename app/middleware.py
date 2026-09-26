import time
import uuid
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)

# IST timezone offset
IST = timezone(timedelta(hours=5, minutes=30))


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request for tracing."""
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter for the /call endpoint."""
    def __init__(self, app, max_per_minute: int = 30):
        super().__init__(app)
        self.max_per_minute = max_per_minute
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path == "/call":
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            # Clean old entries
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] if now - t < 60
            ]
            if len(self._requests[client_ip]) >= self.max_per_minute:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return Response(
                    content='{"detail": "Rate limit exceeded. Try again later."}',
                    status_code=429,
                    media_type="application/json"
                )
            self._requests[client_ip].append(now)
        return await call_next(request)


def is_within_calling_hours() -> bool:
    """Check if current IST time is within TRAI-compliant calling hours (9 AM - 9 PM)."""
    if not settings.ENFORCE_CALLING_HOURS:
        return True
    now_ist = datetime.now(IST)
    return settings.CALLING_HOURS_START <= now_ist.hour < settings.CALLING_HOURS_END


def get_next_calling_window() -> str:
    """Get human-readable next available calling time."""
    now_ist = datetime.now(IST)
    if now_ist.hour >= settings.CALLING_HOURS_END:
        next_time = now_ist.replace(hour=settings.CALLING_HOURS_START, minute=0, second=0) + timedelta(days=1)
    else:
        next_time = now_ist.replace(hour=settings.CALLING_HOURS_START, minute=0, second=0)
    return next_time.strftime("%I:%M %p IST, %d %b %Y")
