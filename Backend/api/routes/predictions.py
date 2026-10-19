from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from Backend.schemas.prediction import (
    MarketPredictionsResponse,
    PredictionSnapshot,
)
from Backend.services.prediction_service import prediction_service

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("", response_model=MarketPredictionsResponse)
def get_all_predictions(
    direction: Optional[str] = Query(default=None, description="Filter by direction: UP or DOWN"),
    min_probability: Optional[float] = Query(default=None, ge=0.0, le=100.0, description="Minimum UP probability %"),
    max_probability: Optional[float] = Query(default=None, ge=0.0, le=100.0, description="Maximum UP probability %"),
    min_return: Optional[float] = Query(default=None, description="Minimum predicted return %"),
    max_return: Optional[float] = Query(default=None, description="Maximum predicted return %"),
    confidence: Optional[str] = Query(default=None, description="Filter by confidence: High or Low"),
    search: Optional[str] = Query(default=None, description="Search ticker or company name"),
    sort_by: str = Query(default="probability_up", description="Field to sort by: probability_up, predicted_return_pct, ticker, company_name, latest_close, predicted_price_5d"),
    sort_order: str = Query(default="desc", description="Sort order: asc or desc"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=500, description="Items per page"),
):
    try:
        result = prediction_service.get_market_predictions(
            direction=direction,
            min_probability=min_probability,
            max_probability=max_probability,
            min_return=min_return,
            max_return=max_return,
            confidence=confidence,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
        )
        return MarketPredictionsResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market predictions: {str(exc)}") from exc


@router.get("/{ticker}", response_model=PredictionSnapshot)
def get_prediction_for_ticker(ticker: str):
    ticker_key = ticker.strip().upper()
    try:
        prediction = prediction_service.get_prediction_for_ticker(ticker_key)
        return PredictionSnapshot(**prediction)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate prediction: {str(exc)}") from exc
