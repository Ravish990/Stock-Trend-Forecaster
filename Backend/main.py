from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Backend.api.routes.market import router as market_router
from Backend.api.routes.predictions import router as predictions_router
from Backend.api.routes.stocks import router as stocks_router
from Backend.services.prediction_service import prediction_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stock_trend_forecaster")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Stock Trend Forecaster backend...")
    try:
        # Precompute & cache predictions in memory for all 471 valid tickers
        prediction_service.warmup_all_predictions()
        logger.info(
            f"Model and dataset ready. Cached {len(prediction_service._prediction_cache)} ticker predictions."
        )
    except Exception as exc:
        logger.error(f"Error during prediction service warmup: {exc}", exc_info=True)
    yield
    logger.info("Shutting down Stock Trend Forecaster backend.")


app = FastAPI(
    title="Stock Trend Forecaster API",
    description="Full-stack 5-day stock price forecasting and directional market probability engine",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks_router)
app.include_router(predictions_router)
app.include_router(market_router)


@app.get("/api/health", tags=["system"])
def health_check():
    is_loaded = prediction_service.model is not None and prediction_service.scaler is not None
    cached_count = len(prediction_service._prediction_cache)
    dataset_avail = (
        prediction_service._dataset_cache is not None
        or cached_count > 0
        or prediction_service.dataset_path.exists()
    )
    return {
        "status": "ok",
        "model_loaded": is_loaded,
        "dataset_available": dataset_avail,
        "latest_data_date": prediction_service.latest_data_date,
        "total_supported_tickers": 471,
        "cached_predictions_count": cached_count,
        "missing_tickers_count": 0,
        "model_name": prediction_service.model_name,
        "forecast_horizon": "5 trading days",
    }


@app.get("/", tags=["system"])
def root():
    return {
        "name": "Stock Trend Forecaster API",
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs",
    }


@app.post("/api/admin/reload-cache", tags=["system"])
@app.get("/api/admin/reload-cache", tags=["system"])
def reload_cache():
    """Reload persistent caches and refresh in-memory predictions."""
    loaded = prediction_service._load_persistent_caches()
    if not loaded:
        prediction_service._prediction_cache.clear()
        prediction_service.warmup_all_predictions()
    return {
        "status": "success",
        "message": "Cache successfully reloaded into memory",
        "model_name": prediction_service.model_name,
        "predicted_up_count": sum(1 for p in prediction_service._prediction_cache.values() if p["predicted_direction"] == "UP"),
        "predicted_down_count": sum(1 for p in prediction_service._prediction_cache.values() if p["predicted_direction"] == "DOWN"),
        "cached_tickers_count": len(prediction_service._prediction_cache),
    }
