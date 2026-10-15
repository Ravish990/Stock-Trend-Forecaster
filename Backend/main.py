import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Data_Processing.target_data import get_data as get_stock_data
from Model.model import LSTMModel
from Model.train_model import FEATURES, HORIZON, SEQUENCE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "best_lstm_stock_model_v5_scaled.pth")
SCALER_PATH = os.path.join(ROOT, "lstm_stock_scalers_v5_scaled.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Stock Trend Forecaster")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model & scaler once at startup ---
scaler = joblib.load(SCALER_PATH)
model = LSTMModel(input_size=len(FEATURES), hidden_size=32, num_layers=2, dropout=0.5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

data_cache: pd.DataFrame | None = None


def get_data() -> pd.DataFrame:
    global data_cache
    if data_cache is None:
        import os
        os.chdir(ROOT)  # target_data reads SP500_Historical_Data_cleaned.csv relative to root
        data_cache = get_stock_data(horizon=HORIZON)
    return data_cache


class PredictRequest(BaseModel):
    ticker: str
    date: str  # YYYY-MM-DD — use latest available if omitted or "latest"


@app.get("/tickers")
def list_tickers():
    df = get_data()
    return sorted(df["Ticker"].dropna().unique().tolist())


@app.post("/predict")
def predict(req: PredictRequest):
    df = get_data()
    ticker = req.ticker.upper()

    stock_df = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    if stock_df.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")

    requested_date = None
    if req.date and req.date.lower() != "latest":
        requested_date = pd.Timestamp(req.date)
        # Check if requested date is a valid NYSE trading day
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=requested_date - pd.Timedelta(days=1),
            end_date=requested_date,
        )
        is_trading_day = (not schedule.empty) and any(
            d.date() == requested_date.date()
            for d in pd.DatetimeIndex(schedule.index)
        )
        hist = stock_df[stock_df["Date"] <= requested_date]
    else:
        requested_date = None
        is_trading_day = True
        hist = stock_df

    if hist.empty:
        raise HTTPException(status_code=400, detail=f"No data for '{ticker}' on or before {requested_date.date() if requested_date else 'latest'}.")

    actual_date = hist["Date"].iloc[-1]
    market_closed_warning = None
    if requested_date is not None and not is_trading_day:
        market_closed_warning = f"Market was closed on {requested_date.date()} (weekend or US holiday). Using last trading date: {actual_date.date()}." 

    if len(hist) < SEQUENCE:
        raise HTTPException(status_code=400, detail=f"Not enough history for '{ticker}' (need {SEQUENCE} rows).")

    missing_features = [f for f in FEATURES if f not in hist.columns]
    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

    recent = hist.tail(SEQUENCE)[FEATURES].copy()
    if recent.isnull().any().any():
        recent = recent.ffill().bfill()

    try:
        scaled = scaler.transform(recent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaling error: {e}")

    X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_ret_log, pred_dir_logit = model(X)

    prob_up = torch.sigmoid(pred_dir_logit).item()
    pred_log_return = pred_ret_log.item()
    current_price = float(hist["Close"].iloc[-1])
    predicted_price = current_price * np.exp(pred_log_return)
    direction = "Up" if prob_up >= 0.5 else "Down"

    # Target date: HORIZON NYSE trading days out (skips weekends + US holidays)
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=actual_date + pd.Timedelta(days=1),
        end_date=actual_date + pd.Timedelta(days=HORIZON * 3),
    )
    if len(schedule) < HORIZON:
        raise HTTPException(status_code=500, detail="Could not compute target trading date.")
    target_date = schedule.index[HORIZON - 1].date()

    response = {
        "ticker": ticker,
        "as_of_date": str(actual_date.date()),
        "target_date": str(target_date),
        "horizon_days": HORIZON,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_log_return": round(pred_log_return * 100, 4),
        "prob_up": round(prob_up * 100, 2),
        "direction": direction,
    }
    if market_closed_warning:
        response["warning"] = market_closed_warning
    return response


@app.get("/health")
def health():
    return {"status": "ok"}
