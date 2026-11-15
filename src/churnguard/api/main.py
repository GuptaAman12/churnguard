"""FastAPI prediction service with observability and model hot-reload."""

from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

import pandas as pd
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from churnguard.api import metrics
from churnguard.api.model_loader import ModelNotAvailableError, ModelService
from churnguard.api.prediction_log import PredictionLogger
from churnguard.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    DriftFeature,
    DriftSummaryResponse,
    ExplanationResponse,
    HealthResponse,
    ModelDescriptor,
    PredictionResponse,
    ReadinessResponse,
)
from churnguard.config import Settings, get_settings
from churnguard.logging import configure_logging, get_logger
from churnguard.monitoring.summary import DriftSummary, load_summary
from churnguard.scoring import risk_tier

logger = get_logger(__name__)


def _require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    settings: Settings = request.app.state.settings
    if not settings.api_key:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )


def _service(request: Request) -> ModelService:
    service: ModelService = request.app.state.model_service
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet; retry shortly",
            headers={"Retry-After": "10"},
        )
    return service


def _prediction_payload(
    *,
    features: CustomerFeatures,
    probability: float,
    service: ModelService,
    descriptor: dict[str, Any],
) -> PredictionResponse:
    threshold = float(descriptor["threshold"])
    return PredictionResponse(
        prediction_id=str(uuid.uuid4()),
        churn_probability=round(probability, 6),
        churn_prediction=bool(probability >= threshold),
        risk_tier=risk_tier(probability, threshold),
        thresholds={"decision": threshold},
        model=ModelDescriptor(**descriptor),
    )


def _record_prediction(response: PredictionResponse) -> None:
    metrics.PREDICTIONS.labels(
        prediction=str(response.churn_prediction).lower(),
        risk_tier=response.risk_tier,
    ).inc()
    metrics.PREDICTION_PROBABILITY.observe(response.churn_probability)


def _drift_timestamp_epoch(summary: DriftSummary) -> float:
    try:
        parsed = datetime.fromisoformat(summary.generated_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except ValueError:
        return time.time()


def create_app(
    settings: Settings | None = None,
    model_service: ModelService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_directories()
    configure_logging(settings.log_level, settings.log_json)
    service = model_service or ModelService(settings)
    prediction_logger = PredictionLogger(
        settings.prediction_log_path, enabled=settings.prediction_log_enabled
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            await run_in_threadpool(service.load)
        except ModelNotAvailableError as exc:
            logger.warning("initial_model_load_failed", error=str(exc))

        tasks: list[asyncio.Task[None]] = []
        if settings.env != "test":
            tasks = [
                asyncio.create_task(_model_refresh_loop(service, settings)),
                asyncio.create_task(_drift_refresh_loop(settings)),
            ]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            prediction_logger.close()

    app = FastAPI(
        title="ChurnGuard Prediction API",
        description=(
            "Calibrated, explainable customer-churn scoring with an MLflow-backed "
            "model registry, TreeSHAP explanations, Prometheus metrics and drift "
            "monitoring."
        ),
        version=settings.api_version,
        lifespan=lifespan,
        contact={"name": "ChurnGuard", "url": "https://github.com/GuptaAman12/churnguard"},
        license_info={"name": "MIT"},
    )
    app.state.settings = settings
    app.state.model_service = service
    app.state.prediction_logger = prediction_logger

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observability(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            endpoint = getattr(route, "path", "unmatched")
            code = str(response.status_code) if response is not None else "500"
            metrics.HTTP_REQUESTS.labels(
                method=request.method, endpoint=endpoint, status=code
            ).inc()
            metrics.HTTP_LATENCY.labels(endpoint=endpoint).observe(elapsed)
            logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                endpoint=endpoint,
                status=int(code),
                duration_ms=round(elapsed * 1000, 2),
            )
            if response is not None:
                response.headers["X-Request-ID"] = request_id
            structlog.contextvars.clear_contextvars()

    @app.get("/", tags=["meta"])
    def root() -> dict[str, Any]:
        return {
            "service": settings.app_name,
            "version": settings.api_version,
            "environment": settings.env,
            "docs": "/docs",
            "metrics": "/metrics",
            "model_info": "/model/info",
            "explain": "/explain",
        }

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=settings.api_version, environment=settings.env)

    @app.get("/ready", response_model=ReadinessResponse, tags=["meta"])
    def ready() -> ReadinessResponse:
        loaded = service.is_loaded
        return ReadinessResponse(
            ready=loaded,
            model_loaded=loaded,
            detail=None if loaded else "No model loaded in registry or fallback path",
        )

    @app.get("/model/info", tags=["model"])
    def model_info(request: Request) -> dict[str, Any]:
        svc = _service(request)
        try:
            return svc.model_metadata()
        except ModelNotAvailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["inference"],
        dependencies=[Depends(_require_api_key)],
    )
    async def predict(payload: CustomerFeatures, request: Request) -> PredictionResponse:
        svc = _service(request)
        descriptor = svc.descriptor()
        frame = payload.to_frame()
        probabilities = await run_in_threadpool(svc.predict_proba, frame)
        response = _prediction_payload(
            features=payload,
            probability=float(probabilities[0]),
            service=svc,
            descriptor=descriptor,
        )
        _record_prediction(response)
        request.app.state.prediction_logger.log(
            payload.model_dump(by_alias=True, mode="json"),
            {
                "prediction_id": response.prediction_id,
                "churn_probability": response.churn_probability,
                "churn_prediction": response.churn_prediction,
                "risk_tier": response.risk_tier,
                "model_version": descriptor.get("version"),
            },
        )
        logger.info(
            "prediction_served",
            prediction_id=response.prediction_id,
            probability=response.churn_probability,
            risk_tier=response.risk_tier,
            model_version=descriptor.get("version"),
        )
        return response

    @app.post(
        "/explain",
        response_model=ExplanationResponse,
        tags=["inference"],
        dependencies=[Depends(_require_api_key)],
    )
    async def explain(payload: CustomerFeatures, request: Request) -> ExplanationResponse:
        svc = _service(request)
        descriptor = svc.descriptor()
        frame = payload.to_frame()
        explanation = await run_in_threadpool(svc.explain, frame)
        response = _prediction_payload(
            features=payload,
            probability=float(explanation["calibrated_probability"]),
            service=svc,
            descriptor=descriptor,
        )
        _record_prediction(response)
        fields = {
            key: value for key, value in explanation.items() if key != "calibrated_probability"
        }
        logger.info(
            "explanation_served",
            prediction_id=response.prediction_id,
            probability=response.churn_probability,
            top_feature=fields["top_features"][0]["feature"] if fields["top_features"] else None,
            model_version=descriptor.get("version"),
        )
        return ExplanationResponse(prediction=response, **fields)

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["inference"],
        dependencies=[Depends(_require_api_key)],
    )
    async def predict_batch(
        payload: BatchPredictionRequest, request: Request
    ) -> BatchPredictionResponse:
        svc = _service(request)
        descriptor = svc.descriptor()
        started = time.perf_counter()
        frame = pd.concat([item.to_frame() for item in payload.customers], ignore_index=True)
        probabilities = await run_in_threadpool(svc.predict_proba, frame)
        predictions = [
            _prediction_payload(
                features=item,
                probability=float(probability),
                service=svc,
                descriptor=descriptor,
            )
            for item, probability in zip(payload.customers, probabilities, strict=True)
        ]
        for prediction in predictions:
            _record_prediction(prediction)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "batch_prediction_served",
            count=len(predictions),
            latency_ms=latency_ms,
            model_version=descriptor.get("version"),
        )
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            latency_ms=latency_ms,
            model=ModelDescriptor(**descriptor),
        )

    @app.post(
        "/admin/reload",
        tags=["admin"],
        dependencies=[Depends(_require_api_key)],
    )
    async def reload_model(request: Request) -> dict[str, Any]:
        try:
            bundle = await run_in_threadpool(request.app.state.model_service.load)
        except ModelNotAvailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "reloaded": True,
            "source": bundle.source,
            "version": bundle.version,
            "loaded_at": bundle.loaded_at,
        }

    @app.get(
        "/monitoring/drift",
        response_model=DriftSummaryResponse,
        tags=["monitoring"],
    )
    def drift(request: Request) -> DriftSummaryResponse:
        summary = load_summary(settings.drift_summary_path)
        if summary is None:
            return DriftSummaryResponse(available=False, threshold=settings.drift_share_threshold)
        metrics.record_drift(
            drift_share=summary.drift_share,
            drifted_features=summary.drifted_features,
            generated_at_epoch=_drift_timestamp_epoch(summary),
        )
        return DriftSummaryResponse(
            available=True,
            generated_at=summary.generated_at,
            dataset_drift=summary.dataset_drift,
            drift_share=summary.drift_share,
            drifted_features=summary.drifted_features,
            total_features=summary.total_features,
            threshold=summary.threshold,
            features=[
                DriftFeature(
                    feature=item.feature,
                    drift_detected=item.drift_detected,
                    drift_score=item.drift_score,
                    method=item.method,
                )
                for item in summary.features
            ],
        )

    @app.get("/metrics", tags=["monitoring"], include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


async def _model_refresh_loop(service: ModelService, settings: Settings) -> None:
    while True:
        try:
            if service.is_loaded:
                changed = await run_in_threadpool(service.refresh_if_changed)
                if changed:
                    logger.info("model_hot_reloaded")
            else:
                await run_in_threadpool(service.load)
        except Exception as exc:  # pragma: no cover - background resilience
            logger.warning("model_refresh_failed", error=str(exc))
        await asyncio.sleep(settings.model_refresh_seconds)


async def _drift_refresh_loop(settings: Settings) -> None:
    while True:
        try:
            summary = await run_in_threadpool(load_summary, settings.drift_summary_path)
            if summary is not None:
                metrics.record_drift(
                    drift_share=summary.drift_share,
                    drifted_features=summary.drifted_features,
                    generated_at_epoch=_drift_timestamp_epoch(summary),
                )
        except Exception as exc:  # pragma: no cover - background resilience
            logger.warning("drift_refresh_failed", error=str(exc))
        await asyncio.sleep(60)


app = create_app()
