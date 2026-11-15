"""Edge-level request protection: per-client rate limiting and body size caps.

Both are deliberately in-process. Pods behind the same load balancer each keep
their own budget, which is the behaviour you want with horizontal scaling: the
limit degrades gracefully instead of depending on a shared store being up. A
cluster-wide quota belongs at the ingress/gateway layer, not here.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


async def _send_json(
    send: Send,
    status: int,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    raw: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("latin-1")),
    ]
    for name, value in (headers or {}).items():
        raw.append((name.lower().encode("latin-1"), str(value).encode("latin-1")))
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send({"type": "http.response.body", "body": body})


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """Token bucket per client key.

    ``per_minute`` sets the sustained rate; ``burst`` is the bucket capacity, so
    a caller can absorb a short spike and then has to pace itself.
    """

    def __init__(self, per_minute: int, burst: int | None = None, *, max_keys: int = 20_000):
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.per_minute = per_minute
        self.rate_per_second = per_minute / 60.0
        self.capacity = float(burst if burst is not None else max(per_minute // 6, 1))
        self.max_keys = max_keys
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def allow(self, key: str) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, updated=now)
                self._buckets[key] = bucket
            else:
                elapsed = now - bucket.updated
                bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.rate_per_second)
                bucket.updated = now
            self._sweep(now)

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0
            missing = (1.0 - bucket.tokens) / self.rate_per_second
            return False, max(1, math.ceil(missing))

    def _sweep(self, now: float) -> None:
        """Drop buckets that have refilled completely; keeps memory bounded."""
        if now - self._last_sweep < 60 and len(self._buckets) <= self.max_keys:
            return
        self._last_sweep = now
        idle = [
            key
            for key, bucket in self._buckets.items()
            if now - bucket.updated > 120 and bucket.tokens >= self.capacity
        ]
        for key in idle:
            del self._buckets[key]
        if len(self._buckets) > self.max_keys:
            # Pathological key churn: forget the oldest half rather than grow.
            oldest = sorted(self._buckets.items(), key=lambda item: item[1].updated)
            for key, _ in oldest[: len(oldest) // 2]:
                del self._buckets[key]


class RateLimitMiddleware:
    """Reject requests over budget with a 429 and a ``Retry-After`` hint."""

    def __init__(
        self,
        app: ASGIApp,
        limiter: TokenBucketLimiter,
        exempt_paths: Iterable[str] = ("/health", "/ready", "/metrics", "/"),
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        key = client[0] if client else "unknown"
        allowed, retry_after = self.limiter.allow(key)
        if not allowed:
            await _send_json(
                send,
                429,
                {
                    "detail": "Rate limit exceeded; retry later",
                    "limit_per_minute": self.limiter.per_minute,
                },
                {"Retry-After": str(retry_after)},
            )
            return

        await self.app(scope, receive, send)


class _BodyTooLargeError(Exception):
    pass


class BodySizeLimitMiddleware:
    """Reject request bodies above ``max_bytes``.

    Checks ``Content-Length`` when present and also counts bytes as the body
    streams, so a chunked request cannot bypass the limit. Only rejects before
    the response has started; once headers are out there is nothing to do but
    stop counting.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > self.max_bytes:
                    await self._reject(send)
                    return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _BodyTooLargeError:
            if not started:
                await self._reject(send)

    async def _reject(self, send: Send) -> None:
        await _send_json(
            send,
            413,
            {"detail": f"Request body exceeds the {self.max_bytes} byte limit"},
            {"Connection": "close"},
        )
