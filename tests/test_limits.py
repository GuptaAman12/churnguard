"""Rate limiting and request-size guard tests."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from churnguard.api.limits import TokenBucketLimiter
from churnguard.api.main import create_app


def test_limiter_allows_a_burst_then_paces() -> None:
    limiter = TokenBucketLimiter(per_minute=60, burst=2)  # 1 token/second
    assert limiter.allow("client-a") == (True, 0)
    assert limiter.allow("client-a") == (True, 0)

    allowed, retry_after = limiter.allow("client-a")
    assert allowed is False
    assert retry_after >= 1

    time.sleep(1.1)
    assert limiter.allow("client-a")[0] is True


def test_limiter_keeps_clients_independent() -> None:
    limiter = TokenBucketLimiter(per_minute=60, burst=1)
    assert limiter.allow("client-a")[0] is True
    assert limiter.allow("client-a")[0] is False
    assert limiter.allow("client-b")[0] is True


def test_limiter_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        TokenBucketLimiter(per_minute=0)


def _limited_app(settings, stub_service, **overrides):
    return create_app(settings.model_copy(update=overrides), stub_service)


def test_rate_limit_returns_429_with_retry_after(settings, stub_service, customer_payload) -> None:
    app = _limited_app(settings, stub_service, rate_limit_per_minute=60, rate_limit_burst=1)
    with TestClient(app) as client:
        assert client.post("/predict", json=customer_payload).status_code == 200

        throttled = client.post("/predict", json=customer_payload)
        assert throttled.status_code == 429
        assert int(throttled.headers["Retry-After"]) >= 1
        assert throttled.json()["limit_per_minute"] == 60


def test_probe_endpoints_bypass_the_rate_limit(settings, stub_service) -> None:
    app = _limited_app(settings, stub_service, rate_limit_per_minute=60, rate_limit_burst=1)
    with TestClient(app) as client:
        for _ in range(3):
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200


def test_body_size_limit_returns_413(settings, stub_service, customer_payload) -> None:
    app = _limited_app(settings, stub_service, max_request_bytes=200)
    with TestClient(app) as client:
        response = client.post("/predict", json=customer_payload)
        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"]


def test_normal_payloads_pass_the_size_guard(settings, stub_service, customer_payload) -> None:
    app = _limited_app(settings, stub_service)
    with TestClient(app) as client:
        assert client.post("/predict", json=customer_payload).status_code == 200
