
import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from tqdm import tqdm

from Model.model import LSTMModel
from Model.train_model import (
    FEATURES,
    MODEL_PATH,
    SCALER_PATH,
    SEQUENCE,
    device,
)

# Keep this equal to the horizon used during model training.
HORIZON = 5

MAX_POSITIONS = 10
MAX_PER_SECTOR = 2
LONG_THRESHOLD = 0.60
TEST_FRACTION = 0.15


def load_model_and_scaler():
    print("Loading global LSTM model and scaler...")

    scaler = joblib.load(SCALER_PATH)

    model = LSTMModel(
        input_size=len(FEATURES),
        hidden_size=32,
        num_layers=2,
        dropout=0.5,
    ).to(device)

    state = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    return model, scaler


def predict_window(model, scaler, window):
    """Predict log return and probability of an upward move."""
    scaled = scaler.transform(window[FEATURES])

    x = torch.tensor(
        scaled, dtype=torch.float32
    ).unsqueeze(0).to(device)

    pred_ret_log, pred_dir_logit = model(x)

    prob_up = torch.sigmoid(pred_dir_logit).item()
    pred_ret = pred_ret_log.item()

    return pred_ret, prob_up


def evaluate_historical_hit_rate(data, model, scaler):
    """
    Evaluate historical forecasts whose actual HORIZON-step outcome
    is already known.

    Uses the latest TEST_FRACTION of unique dates as the evaluation
    period. Ensure this matches the model's actual held-out test split.
    """

    print("\n" + "=" * 85)
    print("HISTORICAL OUT-OF-SAMPLE EVALUATION")
    print("=" * 85)

    dates = np.sort(data["Date"].dropna().unique())

    if len(dates) < 10:
        print("Not enough distinct dates for historical evaluation.")
        return pd.DataFrame()

    cutoff_idx = max(
        1, int(len(dates) * (1 - TEST_FRACTION))
    )
    test_start = pd.Timestamp(dates[cutoff_idx])

    print(f"Evaluation period starts: {test_start.date()}")
    print(f"Forecast horizon: {HORIZON} observations")

    records = []

    with torch.no_grad():
        for ticker, df in tqdm(
            data.groupby("Ticker"),
            desc="Evaluating history",
            unit="stock",
        ):
            df = df.sort_values("Date").reset_index(drop=True)

            if len(df) < SEQUENCE + HORIZON:
                continue

            # Each window uses only features available on its forecast date.
            # Stride by HORIZON to reduce overlapping evaluation windows.
            for i in range(SEQUENCE - 1, len(df) - HORIZON, HORIZON):
                forecast_row = df.iloc[i]

                if forecast_row["Date"] < test_start:
                    continue

                future_row = df.iloc[i + HORIZON]

                current_price = forecast_row["Close"]
                actual_price = future_row["Close"]

                if (
                    pd.isna(current_price)
                    or pd.isna(actual_price)
                    or current_price <= 0
                    or actual_price <= 0
                ):
                    continue

                window = df.iloc[
                    i - SEQUENCE + 1 : i + 1
                ]

                try:
                    pred_ret, prob_up = predict_window(
                        model, scaler, window
                    )
                except (ValueError, KeyError, TypeError):
                    continue

                actual_log_return = np.log(
                    actual_price / current_price
                )

                # Direction is determined by the model's direction head.
                predicted_direction = (
                    "Up" if prob_up >= 0.5 else "Down"
                )
                actual_direction = (
                    "Up" if actual_log_return > 0 else "Down"
                )

                records.append({
                    "Ticker": ticker,
                    "Forecast_Date": forecast_row["Date"],
                    "Actual_Date": future_row["Date"],
                    "Current_Price": current_price,
                    "Predicted_Target": current_price * np.exp(pred_ret),
                    "Actual_Price": actual_price,
                    "Prob_Up": prob_up,
                    "Predicted_Direction": predicted_direction,
                    "Actual_Direction": actual_direction,
                    "Actual_Return_Pct": (
                        np.exp(actual_log_return) - 1
                    ) * 100,
                    "Hit": predicted_direction == actual_direction,
                    "Long_Signal": prob_up > LONG_THRESHOLD,
                })

    results = pd.DataFrame(records)

    if results.empty:
        print("No eligible historical forecasts were found.")
        print("Check the date split, dataset size, and feature columns.")
        return results

    overall_hit_rate = results["Hit"].mean() * 100

    # A long trade wins if its actual future price is higher.
    long_signals = results[results["Long_Signal"]].copy()

    print(f"\nTotal evaluated forecasts: {len(results):,}")
    print(f"Overall directional hit rate: {overall_hit_rate:.2f}%")

    if not long_signals.empty:
        long_hit_rate = (
            long_signals["Actual_Direction"].eq("Up").mean() * 100
        )

        print(f"Long signals (Prob_Up > {LONG_THRESHOLD:.0%}): "
              f"{len(long_signals):,}")
        print(f"Long-signal hit rate: {long_hit_rate:.2f}%")
        print(
            "Mean actual return on long signals: "
            f"{long_signals['Actual_Return_Pct'].mean():.3f}%"
        )
    else:
        print("No historical long signals exceeded the threshold.")

    print("\nMost recent historical forecasts:")
    columns = [
        "Ticker",
        "Forecast_Date",
        "Actual_Date",
        "Current_Price",
        "Predicted_Target",
        "Actual_Price",
        "Prob_Up",
        "Predicted_Direction",
        "Actual_Direction",
        "Hit",
    ]

    print(
        results.sort_values("Actual_Date")
        .tail(20)[columns]
        .to_string(
            index=False,
            formatters={
                "Current_Price": "{:.2f}".format,
                "Predicted_Target": "{:.2f}".format,
                "Actual_Price": "{:.2f}".format,
                "Prob_Up": "{:.2%}".format,
            },
        )
    )

    return results


def run_screener():
    print("Loading cross-sectional market data...")

    data = pd.read_csv("merged_news_target_data.csv")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    data = data.dropna(subset=["Date", "Ticker", "Close"])
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    latest_date = data["Date"].max()
    print(
        f"Screening market based on latest available data: "
        f"{latest_date.date()}"
    )

    model, scaler = load_model_and_scaler()

    # 1. Historical evaluation against known future prices.
    historical = evaluate_historical_hit_rate(
        data, model, scaler
    )

    # 2. Generate current candidate signals.
    trade_signals = []

    grouped = data.groupby("Ticker")

    print(f"\nScanning {len(grouped)} tickers for live candidates...")

    with torch.no_grad():
        for ticker, df in tqdm(
            grouped,
            desc="Processing Tickers",
            unit="stock",
        ):
            df = df.sort_values("Date").reset_index(drop=True)

            if len(df) < SEQUENCE:
                continue

            recent_df = df.tail(SEQUENCE).copy()

            try:
                pred_ret, prob_up = predict_window(
                    model, scaler, recent_df
                )
            except (ValueError, KeyError, TypeError):
                continue

            current_price = float(df["Close"].iloc[-1])

            if current_price <= 0 or not np.isfinite(pred_ret):
                continue

            target_price = current_price * np.exp(pred_ret)

            trade_signals.append({
                "Ticker": ticker,
                "Current_Price": current_price,
                "Target_Price": target_price,
                "Prob_Up": prob_up,
            })

    results_df = pd.DataFrame(trade_signals)

    if results_df.empty:
        print("\nNo valid predictions generated.")
        return

    longs = results_df[
        results_df["Prob_Up"] > LONG_THRESHOLD
    ].sort_values("Prob_Up", ascending=False)

    print("\n" + "=" * 85)
    print("CONSTRUCTING SECTOR-DIVERSIFIED PORTFOLIO")
    print("=" * 85)

    final_portfolio = []
    sector_counts = {}
    sector_cache = {}

    print("Fetching sector data for top candidates from Yahoo Finance...")

    for _, row in longs.iterrows():
        if len(final_portfolio) >= MAX_POSITIONS:
            break

        ticker = row["Ticker"]

        if ticker not in sector_cache:
            try:
                sector_cache[ticker] = (
                    yf.Ticker(ticker).info.get("sector", "Unknown")
                    or "Unknown"
                )
            except Exception:
                sector_cache[ticker] = "Unknown"

        sector = sector_cache[ticker]
        count = sector_counts.get(sector, 0)

        if sector != "Unknown" and count >= MAX_PER_SECTOR:
            continue

        final_portfolio.append({
            "Ticker": ticker,
            "Sector": sector,
            "Conf": row["Prob_Up"],
            "Price": row["Current_Price"],
            "Target": row["Target_Price"],
        })

        if sector != "Unknown":
            sector_counts[sector] = count + 1

    print(
        f"\nTOP DIVERSIFIED LONGS "
        f"(Target: {MAX_POSITIONS}, Max {MAX_PER_SECTOR} per sector)"
    )
    print("-" * 100)

    if final_portfolio:
        for p in final_portfolio:
            expected_change = (
                p["Target"] / p["Price"] - 1
            ) * 100

            print(
                f"{p['Ticker']:<7} | "
                f"{p['Sector']:<25.25} | "
                f"Prob Up: {p['Conf'] * 100:6.2f}% | "
                f"Price: ${p['Price']:8.2f} | "
                f"Target: ${p['Target']:8.2f} | "
                f"Implied change: {expected_change:+7.2f}%"
            )
    else:
        print("No long setups met the strict 60% threshold.")

    print("=" * 100)

    print(
        "\nNote: Live predictions do not have actual future prices yet. "
        "Their hit rate can only be calculated after the forecast horizon "
        "has elapsed and the corresponding price data is available."
    )


if __name__ == "__main__":
    run_screener()
