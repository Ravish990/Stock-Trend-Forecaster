"""
Actual vs Predicted Return — AAPL, February 2026

Evaluates every February trading day using a SEQUENCE-length window.
Plots actual vs predicted percentage returns and labels prediction
confidence as High or Low based on the predicted probability of an
upward move.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from Model.train_model import (
    FEATURES,
    SEQUENCE,
    HORIZON,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    MODEL_PATH,
    device,
    get_data_with_horizon,
    prepare_features,
)
from Model.model import LSTMModel


TICKER = "AAPL"
FEB_START = pd.Timestamp("2026-01-01")
FEB_END = pd.Timestamp("2026-01-")

UPPER_THRESHOLD = 0.60
LOWER_THRESHOLD = 0.40


def run():
    # 1. Load and prepare data
    raw = get_data_with_horizon()

    raw["Date"] = pd.to_datetime(raw["Date"])
    raw_close = raw[["Ticker", "Date", "Close"]].copy()

    data, _ = prepare_features(raw)
    data["Date"] = pd.to_datetime(data["Date"])

    data = data[
        ["Ticker", "Date"] + FEATURES
    ].merge(
        raw_close,
        on=["Ticker", "Date"],
        how="left",
    )

    aapl = (
        data[data["Ticker"] == TICKER]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if aapl.empty:
        print(f"No data found for ticker {TICKER}.")
        return

    # 2. Load trained model
    model = LSTMModel(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    # 3. Generate predictions for February trading days
    records = []

    feb_rows = aapl[
        (aapl["Date"] >= FEB_START)
        & (aapl["Date"] <= FEB_END)
    ]

    with torch.no_grad():
        for _, row in feb_rows.iterrows():
            input_date = row["Date"]
            idx = aapl.index[
                aapl["Date"] == input_date
            ][0]

            if idx < SEQUENCE - 1:
                continue

            window = aapl.iloc[
                idx - SEQUENCE + 1 : idx + 1
            ]

            feat_arr = window[FEATURES].to_numpy(
                dtype=np.float32
            )

            if not np.isfinite(feat_arr).all():
                continue

            X = torch.tensor(
                feat_arr,
                dtype=torch.float32,
            ).unsqueeze(0).to(device)

            pred_ret_log, pred_dir_logit = model(X)

            pred_log_ret = pred_ret_log.item()
            prob_up = torch.sigmoid(
                pred_dir_logit
            ).item()

            current_price = float(row["Close"])

            # Convert predicted log return to simple return (%)
            predicted_return_pct = (
                np.exp(pred_log_ret) - 1
            ) * 100

            predicted_price = (
                current_price * np.exp(pred_log_ret)
            )

            # Direction: always Up or Down
            direction = (
                "Up" if prob_up > 0.50 else "Down"
            )

            # Confidence: based on probability thresholds
            if prob_up > UPPER_THRESHOLD:
                confidence = "High"
            elif prob_up < LOWER_THRESHOLD:
                confidence = "High"
            else:
                confidence = "Low"

            # Actual close HORIZON trading observations later
            future_idx = idx + HORIZON

            if future_idx < len(aapl):
                future_row = aapl.iloc[future_idx]

                actual_future_price = float(
                    future_row["Close"]
                )
                actual_future_date = future_row["Date"]

                actual_return_pct = (
                    actual_future_price / current_price - 1
                ) * 100
            else:
                actual_future_price = np.nan
                actual_future_date = pd.NaT
                actual_return_pct = np.nan

            records.append({
                "Input_Date": input_date,
                "Target_Date": actual_future_date,
                "Current_Price": current_price,
                "Predicted_Price": predicted_price,
                "Actual_Price": actual_future_price,
                "Predicted_Return_%": predicted_return_pct,
                "Actual_Return_%": actual_return_pct,
                "Prob_Up_%": prob_up * 100,
                "Direction": direction,
                "Confidence": confidence,
            })

    df = pd.DataFrame(records)

    if df.empty:
        print("No predictions generated.")
        return

    # 4. Print predictions
    print("\n" + "=" * 125)
    print(
        f"AAPL — ACTUAL vs PREDICTED RETURN | "
        f"February 2026 | Horizon: {HORIZON} trading days"
    )
    print("=" * 125)

    header = (
        f"{'Input Date':<12} | {'Target Date':<12} | "
        f"{'Actual Ret %':>12} | {'Pred Ret %':>11} | "
        f"{'Prob Up':>9} | {'Direction':<9} | "
        f"{'Confidence':<10} | {'Hit?':<5}"
    )

    print(header)
    print("-" * 125)

    hits = 0
    total = 0

    for _, r in df.iterrows():
        actual_ret = r["Actual_Return_%"]

        if pd.isna(actual_ret):
            hit_str = "N/A"
        elif actual_ret == 0:
            hit_str = "N/A"
        else:
            actual_direction = (
                "Up" if actual_ret > 0 else "Down"
            )

            is_hit = (
                r["Direction"] == actual_direction
            )

            hit_str = "YES" if is_hit else "NO"
            hits += int(is_hit)
            total += 1

        target_date = (
            str(r["Target_Date"].date())
            if pd.notna(r["Target_Date"])
            else "N/A"
        )

        actual_text = (
            f"{actual_ret:+.3f}%"
            if pd.notna(actual_ret)
            else "N/A"
        )

        print(
            f"{str(r['Input_Date'].date()):<12} | "
            f"{target_date:<12} | "
            f"{actual_text:>12} | "
            f"{r['Predicted_Return_%']:>+10.3f}% | "
            f"{r['Prob_Up_%']:>8.2f}% | "
            f"{r['Direction']:<9} | "
            f"{r['Confidence']:<10} | "
            f"{hit_str:<5}"
        )

    print("=" * 125)

    if total:
        print(
            f"Directional Hit Rate: {hits}/{total} "
            f"({hits / total * 100:.2f}%)"
        )

    # Confidence breakdown
    print("\nConfidence Breakdown:")

    for level in ["High", "Low"]:
        subset = df[df["Confidence"] == level]
        evaluated = subset[
            subset["Actual_Return_%"].notna()
            & (subset["Actual_Return_%"] != 0)
        ]

        if len(evaluated):
            correct = (
                (
                    (evaluated["Direction"] == "Up")
                    & (evaluated["Actual_Return_%"] > 0)
                )
                |
                (
                    (evaluated["Direction"] == "Down")
                    & (evaluated["Actual_Return_%"] < 0)
                )
            ).sum()

            print(
                f"{level} confidence: "
                f"{correct}/{len(evaluated)} correct "
                f"({correct / len(evaluated) * 100:.2f}%), "
                f"{len(evaluated)} evaluated predictions"
            )
        else:
            print(f"{level} confidence: No evaluable predictions")

    print("=" * 125)

    # 5. Plot actual vs predicted return
    plot_df = df.dropna(
        subset=["Actual_Return_%", "Target_Date"]
    ).copy()

    if plot_df.empty:
        print("Not enough future data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        plot_df["Target_Date"],
        plot_df["Actual_Return_%"],
        marker="o",
        label="Actual Return",
        linewidth=2,
    )

    ax.plot(
        plot_df["Target_Date"],
        plot_df["Predicted_Return_%"],
        marker="s",
        linestyle="--",
        label="Predicted Return",
        linewidth=2,
    )

    ax.axhline(
        y=0,
        linewidth=1,
        linestyle=":",
        alpha=0.8,
    )

    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=5)
    )
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d")
    )

    plt.xticks(rotation=45)

    ax.set_title(
        f"AAPL — Actual vs Predicted Return "
        f"(February 2026, {HORIZON}-day horizon)"
    )
    ax.set_xlabel("Target Date")
    ax.set_ylabel("Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = (
        "aapl_feb2026_actual_vs_predicted_return.png"
    )

    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    run()
