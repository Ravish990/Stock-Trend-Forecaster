import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Data_Processing.target_data import get_data
from Model.model import LSTMModel
from Model.train_model import (
    DROPOUT,
    HIDDEN_SIZE,
    HORIZON,
    MODEL_PATH,
    NUM_LAYERS,
    SCALER_PATH,
)

PLOT_TICKER = "AAPL"

INPUT_DATE = "2026-02-18"

# Number of past days used for the one-day-ahead backtest.
BACKTEST_DAYS = 60


FEATURE_CLIP = 5.0  

# Safety bounds on the predicted daily log return.
RETURN_CLIP = 0.05


SHRINK = 1.0

GRAPH_PATH = "one_day_ahead_backtest.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("ONE-DAY-AHEAD STOCK PREDICTION (NO RECURSION)")
print("=" * 70)
print("Using device:", device)

with open(SCALER_PATH, "rb") as f:
    bundle = pickle.load(f)

feature_scaler = bundle["feature_scaler"]
target_scaler = bundle["target_scaler"]
features = bundle["features"]
sequence_length = bundle["sequence_length"]

print(f"Features: {len(features)} | Sequence length: {sequence_length}")

model = LSTMModel(
    input_size=len(features),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()


print("\nLoading dataset...")
df = get_data(horizon=HORIZON)

required = features + [
    "Date", "Ticker", "Close",
    "next_date", "next_close", "next_log_return", "direction_target",
]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required).copy()
df["Date"] = pd.to_datetime(df["Date"])

ticker_df = (
    df[df["Ticker"] == PLOT_TICKER]
    .sort_values("Date")
    .reset_index(drop=True)
)

if ticker_df.empty:
    raise ValueError(f"No data found for ticker {PLOT_TICKER}")

input_date = pd.Timestamp(INPUT_DATE)
ticker_df = ticker_df[ticker_df["Date"] <= input_date].reset_index(drop=True)

if ticker_df.empty:
    raise ValueError(f"No data on or before {INPUT_DATE}")

if ticker_df["Date"].iloc[-1] != input_date:
    raise ValueError(
        f"{INPUT_DATE} not in dataset. Last available date is "
        f"{ticker_df['Date'].iloc[-1].date()}"
    )
X_all = feature_scaler.transform(ticker_df[features]).astype(np.float32)

if FEATURE_CLIP is not None:
    X_all = np.clip(X_all, -FEATURE_CLIP, FEATURE_CLIP)

print(f"Max |scaled feature| in this ticker: {np.abs(X_all).max():.2f}")
print(f"Max |scaled feature| in last window: "
      f"{np.abs(X_all[-sequence_length:]).max():.2f}"
      "   (values far above ~5 mean off-distribution inputs)")


def predict_rows(row_indexes):
    windows = np.stack([
        X_all[i - sequence_length + 1: i + 1] for i in row_indexes
    ])
    X = torch.tensor(windows, dtype=torch.float32, device=device)

    with torch.no_grad():
        return_scaled, direction_logit = model(X)

    pred_scaled = return_scaled.squeeze(1).cpu().numpy()
    prob_up = torch.sigmoid(direction_logit.squeeze(1)).cpu().numpy()

    pred_log_return = target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()

    pred_log_return = np.clip(pred_log_return * SHRINK, -RETURN_CLIP, RETURN_CLIP)
    return pred_scaled, pred_log_return, prob_up


last_idx = len(ticker_df) - 1
if last_idx < sequence_length - 1:
    raise ValueError("Not enough history for one full sequence.")

pred_scaled, pred_lr, prob_up = predict_rows([last_idx])
row = ticker_df.iloc[last_idx]
current_close = float(row["Close"])
pred_close = current_close * float(np.exp(pred_lr[0]))

print("\n" + "=" * 70)
print("NEXT-DAY PREDICTION")
print("=" * 70)
print(f"Input date            : {row['Date'].date()}")
print(f"Target date           : {pd.Timestamp(row['next_date']).date()}")
print(f"Current close         : {current_close:.4f}")
print(f"Raw scaled output     : {pred_scaled[0]:.4f}  (z-score, NOT a return)")
print(f"Predicted log return  : {pred_lr[0]:.6f}  ({(np.exp(pred_lr[0]) - 1) * 100:.3f}%)")
print(f"Predicted close       : {pred_close:.4f}")
print(f"Direction probability : {prob_up[0]:.4f} -> "
      f"{'UP' if prob_up[0] >= 0.5 else 'DOWN'}")
print(f"Actual next close     : {float(row['next_close']):.4f}")



start = max(sequence_length - 1, len(ticker_df) - BACKTEST_DAYS)
idx = list(range(start, len(ticker_df)))

_, bt_lr, bt_prob = predict_rows(idx)
bt = ticker_df.iloc[idx].copy()

bt["pred_log_return"] = bt_lr
bt["pred_close"] = bt["Close"] * np.exp(bt_lr)
bt["prob_up"] = bt_prob
bt["pred_dir_up"] = bt_prob >= 0.5
bt["actual_dir_up"] = bt["direction_target"] > 0.5
bt["actual_log_return"] = bt["next_log_return"]

# Metrics vs the naive "tomorrow = today" baseline
model_mape = np.mean(np.abs(bt["pred_close"] - bt["next_close"]) / bt["next_close"]) * 100
naive_mape = np.mean(np.abs(bt["Close"] - bt["next_close"]) / bt["next_close"]) * 100

act = bt["actual_log_return"].to_numpy()
prd = bt["pred_log_return"].to_numpy()
r2_vs_zero = 1 - np.sum((act - prd) ** 2) / np.sum(act ** 2)

dir_acc = (bt["pred_dir_up"] == bt["actual_dir_up"]).mean() * 100
majority = max(bt["actual_dir_up"].mean(), 1 - bt["actual_dir_up"].mean()) * 100

print("\n" + "=" * 70)
print(f"BACKTEST: last {len(bt)} days, one-day-ahead, actual inputs only")
print("=" * 70)
print(f"Price MAPE            : {model_mape:.3f}%   (naive: {naive_mape:.3f}%)")
print(f"Return R^2 vs zero    : {r2_vs_zero:.4f}   (>0 beats 'unchanged')")
print(f"Direction accuracy    : {dir_acc:.2f}%   (always-majority: {majority:.2f}%)")
print(f"Pred return std       : {prd.std():.5f}   (actual std: {act.std():.5f})")
print(f"Max |pred return|     : {np.abs(prd).max():.5f}")

print("\nLast 10 rows:")
print(
    bt[["Date", "Close", "pred_close", "next_close", "prob_up"]]
    .tail(10)
    .rename(columns={"Close": "current", "next_close": "actual_next"})
    .to_string(index=False)
)


plt.figure(figsize=(14, 7))
plt.plot(bt["next_date"], bt["next_close"], label="Actual next-day close", linewidth=2)
plt.plot(bt["next_date"], bt["pred_close"], "--", label="Model prediction", linewidth=2)
plt.plot(bt["next_date"], bt["Close"], ":", label="Naive (today's close)", linewidth=1.5)
plt.title(f"{PLOT_TICKER}: one-day-ahead predictions (each uses actual data only)")
plt.xlabel("Target date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(GRAPH_PATH, dpi=150, bbox_inches="tight")
print(f"\nGraph saved as: {GRAPH_PATH}")
plt.show()
