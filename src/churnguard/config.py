"""Typed, environment-driven configuration (12-factor)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, overridable with ``CHURNGUARD_<NAME>`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="CHURNGUARD_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- runtime -----------------------------------------------------------
    env: str = "dev"
    app_name: str = "churnguard"
    api_version: str = "1.0.0"
    api_key: str | None = Field(
        default=None,
        description="When set, mutating/predict endpoints require the X-API-Key header.",
    )
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- paths -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
    train_path: Path = PROJECT_ROOT / "data" / "processed" / "train.parquet"
    test_path: Path = PROJECT_ROOT / "data" / "processed" / "test.parquet"
    reference_path: Path = PROJECT_ROOT / "data" / "reference" / "reference.parquet"
    current_data_path: Path = PROJECT_ROOT / "data" / "current" / "batch.parquet"
    models_dir: Path = PROJECT_ROOT / "models"
    fallback_model_path: Path = PROJECT_ROOT / "models" / "model.joblib"
    reports_dir: Path = PROJECT_ROOT / "reports"
    drift_summary_path: Path = PROJECT_ROOT / "reports" / "drift" / "latest.json"
    prediction_log_path: Path = PROJECT_ROOT / "logs" / "predictions.jsonl"

    # --- MLflow ------------------------------------------------------------
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "churnguard/churn-classification"
    registry_model_name: str = "churnguard-churn-classifier"
    champion_alias: str = "champion"
    model_uri: str | None = None
    model_refresh_seconds: int = 60

    # --- training / promotion ---------------------------------------------
    test_size: float = 0.2
    random_state: int = 42
    decision_threshold: float = 0.5
    min_roc_auc: float = 0.80
    promotion_tolerance: float = 0.005
    optuna_trials: int = 25
    optuna_timeout_seconds: int = 300
    cv_folds: int = 5

    # --- monitoring --------------------------------------------------------
    drift_share_threshold: float = 0.30
    drift_sample_size: int = 2000
    pushgateway_url: str | None = None
    prediction_log_enabled: bool = True

    # --- object storage ----------------------------------------------------
    s3_endpoint_url: str | None = None
    s3_bucket: str = "mlflow"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    def ensure_directories(self) -> None:
        """Create runtime directories; safe to call repeatedly."""
        for path in (
            self.data_dir,
            self.models_dir,
            self.reports_dir,
            self.drift_summary_path.parent,
            self.prediction_log_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
