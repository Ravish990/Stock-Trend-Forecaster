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
import sys
from Backend.services.company_mapping import CompanyMappingService
from Model.model import LSTMModel, AttentionLSTMModel, CrossSectionalScaler
sys.modules['__main__'].CrossSectionalScaler = CrossSectionalScaler
from Model.train_model import FEATURES, SEQUENCE, STOCK_FEATURES, MACRO_FEATURES

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

# 3. Model & Scaler Loading with Dynamic Architecture Detection
scaler = joblib.load(SCALER_PATH)
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

if "attention.attn.0.weight" in state:
    hidden_size = state["attention.attn.0.weight"].shape[1]
    model = AttentionLSTMModel(
        input_size=len(FEATURES),
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.25,
    )
    print(f"Loaded AttentionLSTMModel (hidden_size={hidden_size}) from {MODEL_PATH.name}")
else:
    hidden_size = state["lstm.weight_ih_l0"].shape[0] // 4
    model = LSTMModel(
        input_size=len(FEATURES),
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.5,
    )
    print(f"Loaded LSTMModel (hidden_size={hidden_size}) from {MODEL_PATH.name}")

model.load_state_dict(state)
model.eval()

# 4. Batch predictions
valid_tickers = []
windows = []
latest_rows = []
missing_tickers = []
history_cache = {}

grouped = df.groupby("Ticker", sort=True)
for ticker, group in grouped:
    if ticker == "SNDK":
        continue
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

# Feature normalization
stacked = np.stack(windows)
N, S, F = stacked.shape

is_alpha_model = hasattr(scaler, "macro_scaler")

if is_alpha_model:
    # 1. Standardize stock features cross-sectionally across tickers for each sequence step
    stock_idx = [FEATURES.index(f) for f in STOCK_FEATURES]
    for step in range(S):
        step_vals = stacked[:, step, stock_idx]
        m = np.nanmean(step_vals, axis=0, keepdims=True)
        s = np.nanstd(step_vals, axis=0, keepdims=True) + 1e-6
        stacked[:, step, stock_idx] = np.clip((step_vals - m) / s, -5.0, 5.0)

    # 2. Scale macro features using macro_scaler
    macro_idx = [FEATURES.index(f) for f in MACRO_FEATURES]
    reshaped_macro = stacked[:, :, macro_idx].reshape(N * S, len(macro_idx))
    scaled_macro = scaler.macro_scaler.transform(reshaped_macro)
    stacked[:, :, macro_idx] = np.clip(scaled_macro.reshape(N, S, len(macro_idx)), -5.0, 5.0)
    scaled_windows = stacked
else:
    reshaped = stacked.reshape(N * S, F)
    scaled_reshaped = scaler.transform(reshaped)
    scaled_windows = scaled_reshaped.reshape(N, S, F)

batch_tensor = torch.tensor(scaled_windows, dtype=torch.float32)
with torch.no_grad():
    ret_batch, dir_batch = model(batch_tensor)

raw_logits = dir_batch.squeeze().cpu().numpy()
raw_rets = ret_batch.squeeze().cpu().numpy()

if is_alpha_model:
    # Calibrated Alpha Return Reconstruction:
    # Total expected return = market drift + calibrated alpha
    market_drift = 0.0043  # +0.43% 5-day market baseline
    alpha_std = np.std(raw_rets) + 1e-6
    calibrated_alphas = (raw_rets - np.mean(raw_rets)) / alpha_std * 0.028
    pred_total_rets = market_drift + calibrated_alphas
    
    # Calibrated probability corresponding to expected return
    z_scores = pred_total_rets / 0.035
    prob_ups = 1.0 / (1.0 + np.exp(-z_scores * 1.5))
else:
    prob_ups = 1.0 / (1.0 + np.exp(-raw_logits))
    pred_total_rets = raw_rets

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
    log_ret = float(pred_total_rets[i])
    prob_up = float(prob_ups[i] * 100.0)

    # Predicted price compared directly to current close:
    # If predicted price > latest_close, stock beats current price -> UP
    # If predicted price <= latest_close, stock drops -> DOWN
    pred_price = float(latest_close * np.exp(log_ret))
    pred_return_pct = float(((pred_price / latest_close) - 1.0) * 100.0)

    if pred_return_pct > 0:
        direction = "UP"
        prob_up = max(50.1, prob_up)
    else:
        direction = "DOWN"
        prob_up = min(49.9, prob_up)

    prob_down = float(max(0.0, 100.0 - prob_up))
    confidence = classify_confidence(prob_up)

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
        "model_name": MODEL_PATH.name,
    }

preds_list = list(predictions.values())
up_count = sum(1 for p in preds_list if p["predicted_direction"] == "UP")
down_count = sum(1 for p in preds_list if p["predicted_direction"] == "DOWN")
avg_ret = float(np.mean([p["predicted_return_pct"] for p in preds_list]))
highest_pred = max(preds_list, key=lambda x: x["probability_up"])
lowest_pred = min(preds_list, key=lambda x: x["probability_up"])

summary = {
    "total_supported_tickers": 471,
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
    "model_name": MODEL_PATH.name,
    "forecast_horizon": 5,
}

print(f"\nCache Summary:")
print(f"Total Tickers: {len(predictions)}")
print(f"Predicted UP: {up_count} ({up_count/len(predictions)*100:.1f}%)")
print(f"Predicted DOWN: {down_count} ({down_count/len(predictions)*100:.1f}%)")
print(f"Avg Predicted Return: {avg_ret:.2f}%")

cache_dir = Path("data")
cache_dir.mkdir(exist_ok=True)

with open(cache_dir / "predictions_cache.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2)

with open(cache_dir / "stock_history_cache.json", "w", encoding="utf-8") as f:
    json.dump(history_cache, f)

with open(cache_dir / "market_summary_cache.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(cache_dir / "missing_tickers_cache.json", "w", encoding="utf-8") as f:
    json.dump(missing_tickers, f, indent=2)

print(f"All caches successfully generated and saved to {cache_dir}/ in {time.time()-t0:.2f}s!")
