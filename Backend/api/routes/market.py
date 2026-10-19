from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from Backend.schemas.prediction import (
    LeaderboardResponse,
    MarketSummaryResponse,
    ModelPerformanceResponse,
)
from Backend.schemas.stock import ValidationReportResponse
from Backend.services.prediction_service import prediction_service

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/summary", response_model=MarketSummaryResponse)
def get_market_summary():
    try:
        summary = prediction_service.get_market_summary()
        return MarketSummaryResponse(**summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market summary: {str(exc)}") from exc


@router.get("/leaderboard", response_model=LeaderboardResponse)
def get_market_leaderboard(
    category: str = Query(
        default="highest_probability_up",
        description="Leaderboard category: highest_probability_up, highest_expected_return, highest_probability_down, lowest_expected_return, high_confidence_up, high_confidence_down",
    ),
    limit: int = Query(default=20, ge=1, le=100, description="Number of entries to return"),
):
    try:
        leaderboard = prediction_service.get_leaderboard(category=category, limit=limit)
        return LeaderboardResponse(**leaderboard)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch leaderboard: {str(exc)}") from exc


@router.get("/model-performance", response_model=ModelPerformanceResponse)
def get_model_performance():
    try:
        performance = prediction_service.get_model_performance()
        return ModelPerformanceResponse(**performance)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model performance: {str(exc)}") from exc


@router.get("/validation-report", response_model=ValidationReportResponse)
def get_validation_report():
    try:
        report = prediction_service.get_validation_report()
        return ValidationReportResponse(**report)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch validation report: {str(exc)}") from exc

