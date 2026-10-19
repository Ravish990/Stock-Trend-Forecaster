import math
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionSnapshot(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    latest_close: Optional[float] = None
    latest_data_date: Optional[str] = None
    predicted_price_5d: Optional[float] = None
    predicted_return_pct: Optional[float] = None
    probability_up: Optional[float] = None
    probability_down: Optional[float] = None
    predicted_direction: Optional[str] = None
    confidence_label: Optional[str] = None
    prediction_date: Optional[str] = None
    forecast_horizon: int = 5
    model_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

    @field_validator(
        "company_name",
        "sector",
        "industry",
        "latest_data_date",
        "predicted_direction",
        "confidence_label",
        "prediction_date",
        "model_name",
        mode="before",
    )
    @classmethod
    def sanitize_strings(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        s = str(v).strip()
        if s.lower() in ("nan", "none", "null", ""):
            return None
        return s


class StockPredictionResponse(PredictionSnapshot):
    pass


class MarketPredictionsResponse(BaseModel):
    items: list[PredictionSnapshot] = Field(default_factory=list)
    total: int
    page: int
    page_size: int
    total_pages: int
    generated_at: Optional[str] = None


class MarketSummaryResponse(BaseModel):
    total_supported_tickers: int
    successfully_predicted_tickers: int
    missing_data_tickers: int
    predicted_up_count: int
    predicted_down_count: int
    avg_predicted_return: float
    highest_predicted_probability: float
    lowest_predicted_probability: float
    highest_probability_ticker: Optional[str] = None
    latest_data_date: str
    prediction_generated_at: str
    model_name: str
    forecast_horizon: int = 5


class LeaderboardResponse(BaseModel):
    category: str
    title: str
    description: str
    items: list[PredictionSnapshot] = Field(default_factory=list)
    total: int
    generated_at: Optional[str] = None


class CalibrationBin(BaseModel):
    bin_range: str
    predicted_prob: float
    actual_freq: float
    count: int


class ModelPerformanceResponse(BaseModel):
    evaluation_period: str
    sample_size: int
    evaluated_tickers_count: int
    model_name: str
    forecast_horizon: str
    directional_accuracy_pct: float
    precision: float
    recall: float
    brier_score: float
    log_loss: float
    rmse_log_return: float
    rmse_return_pct: float
    up_fraction_pct: float
    majority_baseline_pct: float
    calibration: list[CalibrationBin] = Field(default_factory=list)
    description: str
