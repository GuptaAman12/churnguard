"""Pydantic request/response contracts for the prediction API."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from churnguard.constants import RAW_FEATURE_COLUMNS

Gender = Literal["Female", "Male"]
YesNo = Literal["Yes", "No"]
MultiLine = Literal["Yes", "No", "No phone service"]
Internet = Literal["DSL", "Fiber optic", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]
Contract = Literal["Month-to-month", "One year", "Two year"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
RiskTier = Literal["low", "medium", "high"]


class CustomerFeatures(BaseModel):
    """One customer's raw features. Field names follow the model contract."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    gender: Gender
    senior_citizen: int = Field(alias="SeniorCitizen", ge=0, le=1)
    partner: YesNo = Field(alias="Partner")
    dependents: YesNo = Field(alias="Dependents")
    tenure: int = Field(ge=0, le=100)
    phone_service: YesNo = Field(alias="PhoneService")
    multiple_lines: MultiLine = Field(alias="MultipleLines")
    internet_service: Internet = Field(alias="InternetService")
    online_security: InternetAddon = Field(alias="OnlineSecurity")
    online_backup: InternetAddon = Field(alias="OnlineBackup")
    device_protection: InternetAddon = Field(alias="DeviceProtection")
    tech_support: InternetAddon = Field(alias="TechSupport")
    streaming_tv: InternetAddon = Field(alias="StreamingTV")
    streaming_movies: InternetAddon = Field(alias="StreamingMovies")
    contract: Contract = Field(alias="Contract")
    paperless_billing: YesNo = Field(alias="PaperlessBilling")
    payment_method: PaymentMethod = Field(alias="PaymentMethod")
    monthly_charges: float = Field(alias="MonthlyCharges", gt=0, le=10_000)
    total_charges: float = Field(alias="TotalCharges", ge=0, le=1_000_000)

    def to_frame(self) -> pd.DataFrame:
        """Serialize to a single-row dataframe with canonical dataset columns."""
        payload = self.model_dump(by_alias=True)
        return pd.DataFrame([payload], columns=list(RAW_FEATURE_COLUMNS))


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class ModelDescriptor(BaseModel):
    name: str
    version: str | None = None
    alias: str | None = None
    source: Literal["registry", "fallback", "injected"]
    model_uri: str | None = None
    threshold: float
    loaded_at: str


class PredictionResponse(BaseModel):
    prediction_id: str
    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: bool
    risk_tier: RiskTier
    thresholds: dict[str, float]
    model: ModelDescriptor


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    latency_ms: float
    model: ModelDescriptor


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    ready: bool
    model_loaded: bool
    detail: str | None = None


class DriftFeature(BaseModel):
    feature: str
    drift_detected: bool
    drift_score: float | None = None
    method: str | None = None


class DriftSummaryResponse(BaseModel):
    available: bool
    generated_at: str | None = None
    dataset_drift: bool | None = None
    drift_share: float | None = None
    drifted_features: int | None = None
    total_features: int | None = None
    threshold: float
    features: list[DriftFeature] = Field(default_factory=list)
