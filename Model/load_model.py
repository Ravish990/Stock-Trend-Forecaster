import numpy as np
import pandas as pd
import torch
import yfinance as yf
from tqdm import tqdm

from Model.model import LSTMModel
from Model.train_model import (
    DROPOUT,
    FEATURES,
    HIDDEN_SIZE,
    HORIZON,
    MODEL_PATH,
    NUM_LAYERS,
    SEQUENCE,
    device,
    get_data_with_horizon,
    prepare_features,
)

# --- CONFIGURATION ---
EVAL_DATE = pd.Timestamp("2026-02-05")
MAX_POSITIONS = 10
MAX_PER_SECTOR = 2


def run_dated_evaluation():
    print(f"Loading data to evaluate as of: {EVAL_DATE.date()}...")
    raw = get_data_with_horizon()
    # Preserve raw Close before scaling overwrites feature columns
    raw_close = raw[["Ticker", "Date", "Close"]].copy()
    data, _ = prepare_features(raw)
    data = data[["Ticker", "Date"] + FEATURES].copy()
    data = data.merge(raw_close, on=["Ticker", "Date"], how="left")

    # Load model
    model = LSTMModel(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    trade_signals = []
    grouped = data.groupby("Ticker")

    print("Running cross-sectional inference...")
    with torch.no_grad():
        for ticker, df in tqdm(grouped, desc="Processing Tickers", unit="stock"):
            df = df.sort_values("Date").reset_index(drop=True)

            # Restrict history strictly up to EVAL_DATE (no lookahead)
            hist = df[df["Date"] <= EVAL_DATE]
            if len(hist) < SEQUENCE:
                continue

            current_row = hist.iloc[-1]
            if current_row["Date"] != EVAL_DATE:
                continue  # Ticker did not trade on the exact evaluation date

            future_slice = df[df["Date"] > EVAL_DATE].reset_index(drop=True)

            # Require at least HORIZON subsequent observations for ground-truth comparison
            if len(future_slice) < HORIZON:
                continue

            actual_target_row = future_slice.iloc[HORIZON - 1]
            actual_future_price = actual_target_row["Close"]
            actual_future_date = actual_target_row["Date"]

            # Features already scaled by prepare_features — use directly
            recent_arr = hist.tail(SEQUENCE)[FEATURES].to_numpy(dtype=np.float32)
            if not np.isfinite(recent_arr).all():
                continue

            X_tensor = (
                torch.tensor(recent_arr, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            pred_ret_log, pred_dir_logit = model(X_tensor)

            prob_up = torch.sigmoid(pred_dir_logit).item()
            pred_ret = pred_ret_log.item()

            current_price = current_row["Close"]
            target_price = current_price * np.exp(pred_ret)

            trade_signals.append(
                {
                    "Ticker": ticker,
                    "Current_Price": current_price,
                    "Target_Price": target_price,
                    "Prob_Up": prob_up,
                    "Actual_Future_Price": actual_future_price,
                    "Future_Date": actual_future_date.date(),
                }
            )

    results_df = pd.DataFrame(trade_signals)
    if results_df.empty:
        print(f"No qualifying trading data found for date {EVAL_DATE.date()}.")
        return

    print(f"\nTotal signals: {len(results_df)} | Prob_Up range: {results_df['Prob_Up'].min():.3f} - {results_df['Prob_Up'].max():.3f}")

    # Adaptive threshold: start at 60%, fall back to top-10 by confidence
    CONF_THRESHOLD = 0.60
    longs = results_df[results_df["Prob_Up"] > CONF_THRESHOLD].sort_values(
        "Prob_Up", ascending=False
    )
    if longs.empty:
        print(f"No signals above {CONF_THRESHOLD*100:.0f}% — using top-{MAX_POSITIONS} by confidence.")
        longs = results_df.sort_values("Prob_Up", ascending=False).head(MAX_POSITIONS)

    print("\nApplying sector caps and fetching industry data...")
    final_portfolio = []
    sector_counts = {}

    for _, row in longs.iterrows():
        if len(final_portfolio) >= MAX_POSITIONS:
            break

        ticker = row["Ticker"]
        try:
            sector = yf.Ticker(ticker).info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"

        if sector not in sector_counts:
            sector_counts[sector] = 0

        if sector_counts[sector] < MAX_PER_SECTOR or sector == "Unknown":
            final_portfolio.append(
                {
                    "Ticker": ticker,
                    "Sector": sector,
                    "Conf": row["Prob_Up"],
                    "Entry_Price": row["Current_Price"],
                    "Target_Price": row["Target_Price"],
                    "Actual_Exit_Price": row["Actual_Future_Price"],
                    "Exit_Date": row["Future_Date"],
                }
            )
            if sector != "Unknown":
                sector_counts[sector] += 1

    # --- PERFORMANCE OUTPUT ---
    print("\n" + "=" * 115)
    print(
        f"SECTOR-NEUTRAL PERFORMANCE: {EVAL_DATE.date()} -> 5 TRADING DAYS OUT"
    )
    print("=" * 115)
    header = f"{'Ticker':<6} | {'Sector':<22} | {'Conf':<7} | {'Entry':>9} | {'Predicted':>10} | {'Actual Exit':>11} | {'Actual Ret':>10} | {'Hit?':>5}"
    print(header)
    print("-" * 115)

    hits = 0
    total_ret = 0.0

    for p in final_portfolio:
        entry = p["Entry_Price"]
        actual_exit = p["Actual_Exit_Price"]
        actual_return = ((actual_exit - entry) / entry) * 100
        total_ret += actual_return

        is_hit = actual_exit > entry
        if is_hit:
            hits += 1

        hit_str = "YES" if is_hit else "NO"
        print(
            f"{p['Ticker']:<6} | {p['Sector']:<22} | {p['Conf']*100:>5.2f}% | "
            f"${entry:>8.2f} | ${p['Target_Price']:>9.2f} | ${actual_exit:>10.2f} | "
            f"{actual_return:>+9.2f}% | {hit_str:>5}"
        )

    portfolio_size = len(final_portfolio)
    if portfolio_size == 0:
        print("No trades passed sector caps.")
        return
    win_rate = (hits / portfolio_size) * 100
    avg_return = total_ret / portfolio_size
    print("=" * 115)
    print(f"Scorecard: {hits}/{portfolio_size} Wins ({win_rate:.1f}% Hit Rate)")
    print(f"Average 5-Day Holding Period Return: {avg_return:>+6.2f}%")
    print("=" * 115)


if __name__ == "__main__":
    run_dated_evaluation()