"""API contract, auth and observability tests."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from churnguard.api.main import create_app
from churnguard.monitoring.summary import DriftSummary, FeatureDrift, write_summary
from tests.conftest import StubModelService


@pytest.fixture
def client(settings, stub_service):
    app = create_app(settings, stub_service)
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_readiness(client, stub_service) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert stub_service.load_calls == 1


def test_root_and_model_info(client) -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["service"] == "churnguard"

    info = client.get("/model/info")
    assert info.status_code == 200
    assert info.json()["source"] == "injected"
    assert info.json()["metrics"]["roc_auc"] == 0.84


def test_single_prediction_contract(client, customer_payload, settings) -> None:
    response = client.post("/predict", json=customer_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_probability"] == pytest.approx(0.73)
    assert body["churn_prediction"] is True
    assert body["risk_tier"] == "high"
    assert body["thresholds"]["decision"] == 0.5
    assert body["model"]["source"] == "injected"
    assert response.headers["X-Request-ID"]

    log_lines = settings.prediction_log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    record = json.loads(log_lines[0])
    assert record["features"]["gender"] == customer_payload["gender"]
    assert record["prediction"]["churn_probability"] == pytest.approx(0.73)


def test_prediction_validation_errors(client, customer_payload) -> None:
    invalid_senior = {**customer_payload, "SeniorCitizen": 5}
    assert client.post("/predict", json=invalid_senior).status_code == 422

    invalid_contract = {**customer_payload, "Contract": "Lifetime"}
    assert client.post("/predict", json=invalid_contract).status_code == 422

    unexpected = {**customer_payload, "email": "a@b.com"}
    assert client.post("/predict", json=unexpected).status_code == 422


def test_batch_prediction(client, customer_payload) -> None:
    response = client.post(
        "/predict/batch", json={"customers": [customer_payload, customer_payload]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["latency_ms"] >= 0


def test_explanation_endpoint(client, customer_payload) -> None:
    response = client.post("/explain", json=customer_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["churn_probability"] == pytest.approx(0.73)
    assert body["top_features"][0]["feature"] == "Contract_Month-to-month"
    assert body["top_features"][0]["direction"] == "increases"
    assert body["base_value"] == pytest.approx(-1.1)
    assert body["caveat"]


def test_batch_rejects_empty_payload(client) -> None:
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_api_key_enforcement(settings, customer_payload) -> None:
    secured_settings = settings.model_copy(update={"api_key": "s3cret"})
    app = create_app(secured_settings, StubModelService())
    with TestClient(app) as client:
        assert client.post("/predict", json=customer_payload).status_code == 401
        assert client.post("/explain", json=customer_payload).status_code == 401
        assert (
            client.post(
                "/predict", json=customer_payload, headers={"X-API-Key": "wrong"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/predict", json=customer_payload, headers={"X-API-Key": "s3cret"}
            ).status_code
            == 200
        )


def test_metrics_endpoint_exposes_domain_metrics(client, customer_payload) -> None:
    client.post("/predict", json=customer_payload)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "churnguard_predictions_total" in response.text
    assert "churnguard_http_requests_total" in response.text
    assert "churnguard_prediction_probability_bucket" in response.text


def test_drift_endpoint_without_report(client) -> None:
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    assert response.json()["available"] is False


def test_drift_endpoint_with_report(client, settings) -> None:
    summary = DriftSummary(
        generated_at="2026-01-01T00:00:00+00:00",
        reference_rows=100,
        current_rows=100,
        drift_share=0.4,
        drifted_features=2,
        total_features=5,
        dataset_drift=True,
        threshold=0.3,
        features=[
            FeatureDrift(feature="tenure", drift_detected=True, drift_score=0.01),
            FeatureDrift(feature="gender", drift_detected=False, drift_score=0.5),
        ],
    )
    write_summary(summary, settings.drift_summary_path)
    body = client.get("/monitoring/drift").json()
    assert body["available"] is True
    assert body["drift_share"] == 0.4
    assert len(body["features"]) == 2
    assert body["features"][0]["feature"] == "tenure"


def test_unloaded_model_returns_503(settings, customer_payload) -> None:
    app = create_app(settings, StubModelService(loaded=False))
    with TestClient(app) as client:
        assert client.get("/ready").json()["ready"] is False
        assert client.post("/predict", json=customer_payload).status_code == 503
        assert client.post("/explain", json=customer_payload).status_code == 503
        assert client.get("/model/info").status_code == 503


def test_request_id_round_trip(client) -> None:
    response = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert response.headers["X-Request-ID"] == "trace-123"
