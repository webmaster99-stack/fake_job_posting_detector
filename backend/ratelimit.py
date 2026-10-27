"""Per-client request limit, kept in memory.

Fine for one process (the API runs a single uvicorn worker on Render's free tier). The client
is the request's IP; uvicorn sets it from Render's `X-Forwarded-For` (`--proxy-headers`).
"""

import math
import threading
import time
from collections import Counter

from fastapi import HTTPException, Request, status


class RateLimit:
    """At most `limit` requests per client in each fixed `window` (seconds)."""

    def __init__(self, limit: int, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window
        self._lock = threading.Lock()
        self._current = -1
        self._counts: Counter[str] = Counter()

    def __call__(self, request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = int(now // self.window)
        with self._lock:
            if window != self._current:  # a new window: forget every client
                self._current, self._counts = window, Counter()
            self._counts[client] += 1
            allowed = self._counts[client] <= self.limit
        if not allowed:
            retry = math.ceil((window + 1) * self.window - now)
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "rate limit exceeded",
                headers={"Retry-After": str(retry)},
            )
