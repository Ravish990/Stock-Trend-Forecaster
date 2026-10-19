import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib

from Backend.core.config import (
    DATASET_PATH,
    MODEL_PATH,
    SCALER_PATH,
    COMPANY_METADATA_PATH,
    AAPL_COMPARISON_PATH,
)
from Backend.services.company_mapping import CompanyMappingService
from Model.model import LSTMModel
from Model.train_model import FEATURES, SEQUENCE

print("Building persistent cache for instant API responses...")
t0 = time.time()

# 1. Company Mapping
mapping_service = CompanyMappingService()
meta_map = mapping_service.build_mapping()

# 2. Load dataset once
print("Reading merged dataset...")
df = pd.read_csv(DATASET_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Ticker", "Date", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
latest_date_str = df["Date"].max().strftime("%Y-%m-%d")
print(f"Dataset loaded in {time.time()-t0:.2f}s, latest date: {latest_date_str}")

# 3. Model & Scaler
scaler = joblib.load(SCALER_PATH)
model = LSTMModel(input_size=len(FEATURES), hidden_size=32, num_layers=2, dropout=0.5)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

# 4. Batch predictions
valid_tickers = []
windows = []
latest_rows = []
missing_tickers = []
history_cache = {}

grouped = df.groupby("Ticker", sort=True)
for ticker, group in grouped:
    # Save recent 90-day history for fast charts
    recent_90 = group.tail(90)
    history_items = []
    for r in recent_90.to_dict(orient="records"):
        history_items.append({
            "date": pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"),
            "close": float(r["Close"]),
            "open": float(r["Open"]) if "Open" in r and pd.notna(r["Open"]) else None,
            "high": float(r["High"]) if "High" in r and pd.notna(r["High"]) else None,
            "low": float(r["Low"]) if "Low" in r and pd.notna(r["Low"]) else None,
            "volume": float(r["Volume"]) if "Volume" in r and pd.notna(r["Volume"]) else None,
        })
    history_cache[ticker] = history_items

    if len(group) < SEQUENCE:
        missing_tickers.append({
            "ticker": ticker,
            "company_name": meta_map.get(ticker, {}).get("company_name"),
            "sector": meta_map.get(ticker, {}).get("sector"),
            "industry": meta_map.get(ticker, {}).get("industry"),
            "reason": f"Only {len(group)} historical observations available; minimum {SEQUENCE} required.",
        })
        continue

    window = group.iloc[-SEQUENCE:][FEATURES]
    if window.isna().any().any():
        missing_tickers.append({
            "ticker": ticker,
            "company_name": meta_map.get(ticker, {}).get("company_name"),
            "sector": meta_map.get(ticker, {}).get("sector"),
            "industry": meta_map.get(ticker, {}).get("industry"),
            "reason": "Missing feature values in latest sequence window.",
        })
        continue

    arr = window.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        missing_tickers.append({
            "ticker": ticker,
            "company_name": meta_map.get(ticker, {}).get("company_name"),
            "sector": meta_map.get(ticker, {}).get("sector"),
            "industry": meta_map.get(ticker, {}).get("industry"),
            "reason": "Non-finite feature values in latest sequence window.",
        })
        continue

    valid_tickers.append(ticker)
    windows.append(arr)
    latest_rows.append({
        "Close": float(group.iloc[-1]["Close"]),
        "Date": pd.Timestamp(group.iloc[-1]["Date"]).strftime("%Y-%m-%d"),
    })

print(f"Extracted {len(valid_tickers)} valid windows and {len(missing_tickers)} missing tickers.")

# PyTorch batch inference
stacked = np.stack(windows)
N, S, F = stacked.shape
reshaped = stacked.reshape(N * S, F)
scaled_reshaped = scaler.transform(reshaped)
scaled_windows = scaled_reshaped.reshape(N, S, F)

batch_tensor = torch.tensor(scaled_windows, dtype=torch.float32)
with torch.no_grad():
    ret_batch, dir_batch = model(batch_tensor)

raw_logits = dir_batch.squeeze().cpu().numpy()
raw_rets = ret_batch.squeeze().cpu().numpy()

# Calibrate logits with median neutral centering to eliminate majority-class upward bias
logit_median = float(np.median(raw_logits))
calibrated_logits = raw_logits - logit_median
prob_ups = 1.0 / (1.0 + np.exp(-calibrated_logits))

# Calibrate returns consistently around neutral market baseline so return sign matches direction
ret_median = float(np.median(raw_rets))
calibrated_rets = raw_rets - ret_median

now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

predictions = {}
def classify_confidence(p):
    if p > 60.0: return "UP \u2014 High confidence"
    elif p > 50.0: return "UP \u2014 Low confidence"
    elif p == 50.0: return "DOWN \u2014 Low confidence"
    elif p >= 40.0: return "DOWN \u2014 Low confidence"
    else: return "DOWN \u2014 High confidence"

for i, ticker in enumerate(valid_tickers):
    latest_close = latest_rows[i]["Close"]
    latest_date = latest_rows[i]["Date"]
    prob_up = float(prob_ups[i] * 100.0)
    prob_down = float(max(0.0, 100.0 - prob_up))
    direction = "UP" if prob_up > 50.0 else "DOWN"
    confidence = classify_confidence(prob_up)

    log_ret = float(calibrated_rets[i])
    # Enforce strict consistency: DOWN predictions have negative return, UP predictions have positive return
    if direction == "DOWN" and log_ret >= 0:
        log_ret = -0.002
    elif direction == "UP" and log_ret <= 0:
        log_ret = 0.002

    pred_price = float(latest_close * np.exp(log_ret))
    pred_return_pct = float(((pred_price / latest_close) - 1.0) * 100.0)

    meta = meta_map.get(ticker, {})
    c_name = meta.get("company_name")
    c_sec = meta.get("sector")
    c_ind = meta.get("industry")
    predictions[ticker] = {
        "ticker": ticker,
        "company_name": None if pd.isna(c_name) else str(c_name),
        "sector": None if pd.isna(c_sec) else str(c_sec),
        "industry": None if pd.isna(c_ind) else str(c_ind),
        "latest_close": latest_close,
        "latest_data_date": latest_date,
        "predicted_price_5d": pred_price,
        "predicted_return_pct": pred_return_pct,
        "probability_up": prob_up,
        "probability_down": prob_down,
        "predicted_direction": direction,
        "confidence_label": confidence,
        "prediction_date": now_iso,
        "forecast_horizon": 5,
        "model_name": "best_lstm_stock_model_v3_h5.pth",
    }

preds_list = list(predictions.values())
up_count = sum(1 for p in preds_list if p["predicted_direction"] == "UP")
down_count = sum(1 for p in preds_list if p["predicted_direction"] == "DOWN")
avg_ret = float(np.mean([p["predicted_return_pct"] for p in preds_list]))
highest_pred = max(preds_list, key=lambda x: x["probability_up"])
lowest_pred = min(preds_list, key=lambda x: x["probability_up"])

summary = {
    "total_supported_tickers": 472,
    "successfully_predicted_tickers": len(predictions),
    "missing_data_tickers": len(missing_tickers),
    "predicted_up_count": up_count,
    "predicted_down_count": down_count,
    "avg_predicted_return": round(avg_ret, 2),
    "highest_predicted_probability": round(highest_pred["probability_up"], 1),
    "lowest_predicted_probability": round(lowest_pred["probability_up"], 1),
    "highest_probability_ticker": highest_pred["ticker"],
    "latest_data_date": latest_date_str,
    "prediction_generated_at": now_iso,
    "model_name": "best_lstm_stock_model_v3_h5.pth",
    "forecast_horizon": 5,
}

# Save caches
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

with open(data_dir / "predictions_cache.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2)

with open(data_dir / "stock_history_cache.json", "w", encoding="utf-8") as f:
    json.dump(history_cache, f)

with open(data_dir / "market_summary_cache.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(data_dir / "missing_tickers_cache.json", "w", encoding="utf-8") as f:
    json.dump(missing_tickers, f, indent=2)

print("Persistent caches built successfully in data/:")
print(f"  - predictions_cache.json ({len(predictions)} items)")
print(f"  - stock_history_cache.json ({len(history_cache)} tickers)")
print(f"  - market_summary_cache.json (UP: {up_count}, DOWN: {down_count}, AVG: {avg_ret:.2f}%)")
print(f"  - missing_tickers_cache.json ({len(missing_tickers)} items)")
print(f"Total time: {time.time()-t0:.2f}s")

