import math
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
from Backend.schemas.prediction import PredictionSnapshot


class StockMetadata(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    latest_close: Optional[float] = None
    latest_data_date: Optional[str] = None
    predicted_price_5d: Optional[float] = None
    predicted_return_pct: Optional[float] = None
    probability_up: Optional[float] = None
    probability_down: Optional[float] = None
    predicted_direction: Optional[str] = None
    confidence_label: Optional[str] = None
    prediction_available: bool = False

    @field_validator(
        "company_name",
        "sector",
        "industry",
        "latest_data_date",
        "predicted_direction",
        "confidence_label",
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


class StockListResponse(BaseModel):
    items: list[StockMetadata] = Field(default_factory=list)
    total: int
    page: int
    page_size: int
    total_pages: int
    search: Optional[str] = None
    sort_by: str
    sort_order: str = "asc"


class StockHistoryItem(BaseModel):
    date: str
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    predicted_price: Optional[float] = None
    hit: Optional[bool] = None
    is_forecast: bool = False


class StockHistoryResponse(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    latest_close: float
    latest_data_date: str
    history: list[StockHistoryItem] = Field(default_factory=list)
    forecast_point: Optional[StockHistoryItem] = None


class StockDetailResponse(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    latest_close: float
    latest_data_date: str
    prediction: Optional[PredictionSnapshot] = None
    history_summary: list[StockHistoryItem] = Field(default_factory=list)


class MissingTickerDetail(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    reason: str


class ValidationReportResponse(BaseModel):
    total_unique_supported_tickers: int
    successfully_mapped_company_names: int
    missing_mappings: int
    duplicate_ticker_checks: int
    tickers_with_valid_prediction_inputs: int
    missing_data_tickers_count: int
    missing_tickers: list[MissingTickerDetail] = Field(default_factory=list)
