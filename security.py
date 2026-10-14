# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  security.py
#  API-key check and a small in-memory rate limiter
#  for the /audit endpoint.
# ─────────────────────────────────────────────

import hmac
import threading
import time
from collections import deque
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

import config

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """Sliding-window limiter: at most per_minute hits in 60 s and per_hour in 3600 s."""

    def __init__(self, per_minute: int, per_hour: int):
        self.per_minute = per_minute
        self.per_hour = per_hour
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, bucket: str) -> Optional[int]:
        """Record a hit and return None if allowed, else seconds to wait."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(bucket, deque())
            while hits and now - hits[0] > 3600:
                hits.popleft()
            recent = [t for t in hits if now - t <= 60]
            if len(hits) >= self.per_hour:
                return int(3600 - (now - hits[0])) + 1
            if len(recent) >= self.per_minute:
                return int(60 - (now - recent[0])) + 1
            hits.append(now)
            return None


limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE, config.RATE_LIMIT_PER_HOUR)


def require_access(api_key: Optional[str] = Depends(api_key_header)) -> None:
    expected = config.API_KEY
    if expected:
        if not api_key or not hmac.compare_digest(api_key.encode(), expected.encode()):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    # Wrong keys are rejected above and never counted, so they cannot
    # use up the real key's allowance.
    wait = limiter.check("key" if expected else "anonymous")
    if wait is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded, retry later",
            headers={"Retry-After": str(wait)},
        )
