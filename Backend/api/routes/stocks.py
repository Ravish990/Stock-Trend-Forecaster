from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from Backend.schemas.stock import (
    StockDetailResponse,
    StockHistoryResponse,
    StockListResponse,
    StockMetadata,
)
from Backend.services.prediction_service import prediction_service

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("", response_model=StockListResponse)
def list_stocks(
    search: Optional[str] = Query(default=None, description="Search ticker or company name"),
    sector: Optional[str] = Query(default=None, description="Filter by sector"),
    direction: Optional[str] = Query(default=None, description="Filter by predicted direction (UP or DOWN)"),
    confidence: Optional[str] = Query(default=None, description="Filter by confidence label (High or Low)"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=500, description="Items per page"),
    sort_by: str = Query(default="ticker", description="Field to sort by"),
    sort_order: str = Query(default="asc", description="Sort order: asc or desc"),
):
    try:
        result = prediction_service.get_stock_listing(
            search=search,
            sector=sector,
            direction=direction,
            confidence=confidence,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return StockListResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock directory: {str(exc)}") from exc


@router.get("/{ticker}/history", response_model=StockHistoryResponse)
def get_stock_history(
    ticker: str,
    limit: int = Query(default=90, ge=5, le=500, description="Number of historical trading days to return"),
):
    try:
        history = prediction_service.get_stock_history(ticker=ticker, limit=limit)
        return StockHistoryResponse(**history)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock history: {str(exc)}") from exc


@router.get("/{ticker}", response_model=StockDetailResponse)
def get_stock_details(ticker: str):
    ticker_key = ticker.strip().upper()
    try:
        hist_data = prediction_service.get_stock_history(ticker_key, limit=15)
        company = prediction_service.company_mapping.get_company_details(ticker_key)

        pred = None
        try:
            pred = prediction_service.get_prediction_for_ticker(ticker_key)
        except Exception:
            pred = None

        return StockDetailResponse(
            ticker=ticker_key,
            company_name=company.get("company_name") or hist_data.get("company_name"),
            sector=company.get("sector") or hist_data.get("sector"),
            industry=company.get("industry") or hist_data.get("industry"),
            latest_close=hist_data.get("latest_close", 0.0),
            latest_data_date=hist_data.get("latest_data_date", prediction_service.latest_data_date),
            prediction=pred,
            history_summary=hist_data.get("history", []),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock details: {str(exc)}") from exc
