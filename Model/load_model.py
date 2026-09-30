import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Data_Processing.target_data import get_data
from Model.model import LSTMModel

from Model.train_model import (
    add_quant_features,
    add_targets,
    HORIZON,
    SEQUENCE_LENGTH,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    MODEL_PATH,
    SCALER_PATH,
)


# ============================================================
# CONFIG
# ============================================================

PLOT_TICKER = "AAPL"

# Last date whose ACTUAL data the model is allowed to see
INPUT_DATE = "2026-02-18"

# Number of future trading days to predict
FORECAST_DAYS = 5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("5-DAY RECURSIVE STOCK PREDICTION")
print("=" * 70)

print("Using device:", device)


# ============================================================
# LOAD SCALER / FEATURE INFORMATION
# ============================================================

print("\nLoading scaler and feature information...")

with open(SCALER_PATH, "rb") as f:
    bundle = pickle.load(f)

feature_scaler = bundle["feature_scaler"]
target_scaler = bundle["target_scaler"]

features = bundle["features"]
sequence_length = bundle["sequence_length"]

regression_target = bundle["regression_target"]
direction_target = bundle["direction_target"]

print(f"Number of features : {len(features)}")
print(f"Sequence length    : {sequence_length}")
print(f"Regression target  : {regression_target}")
print(f"Direction target   : {direction_target}")


print("\nFeatures used by model:")

for i, feature in enumerate(features, 1):
    print(f"{i:2}. {feature}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = get_data(horizon=HORIZON)

print("Raw data shape:", df.shape)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nAdding quantitative features...")

df = add_quant_features(df)

print("Adding targets...")

df = add_targets(
    df,
    horizon=HORIZON
)


# ============================================================
# CLEAN DATA
# ============================================================

required_columns = features + [
    "Date",
    "Ticker",
    "Close",
    "next_date",
    "next_close",
    "next_return",
    regression_target,
    direction_target,
]

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=required_columns
).copy()

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# ============================================================
# SELECT TICKER
# ============================================================

ticker_df = df[
    df["Ticker"] == PLOT_TICKER
].copy()

ticker_df = ticker_df.sort_values(
    "Date"
).reset_index(drop=True)

print("\nTicker:", PLOT_TICKER)

if len(ticker_df) == 0:
    raise ValueError(
        f"No data found for ticker {PLOT_TICKER}"
    )


# ============================================================
# INPUT DATE
# ============================================================

input_date = pd.Timestamp(INPUT_DATE)


# ============================================================
# GET HISTORY UP TO INPUT DATE
# ============================================================

history = ticker_df[
    ticker_df["Date"] <= input_date
].copy()

history = history.sort_values("Date").reset_index(drop=True)


if len(history) == 0:
    raise ValueError(
        f"No data available on or before {INPUT_DATE}"
    )


actual_input_date = history["Date"].iloc[-1]


if actual_input_date != input_date:

    raise ValueError(
        f"{INPUT_DATE} is not available in dataset. "
        f"Last available date is "
        f"{actual_input_date.date()}"
    )


# ============================================================
# APPLY SAVED SCALER
#
# IMPORTANT:
# We don't modify original prices.
# We create scaled feature columns separately.
# ============================================================

print("\nPreparing model features...")

history_scaled = history.copy()

history_scaled.loc[:, features] = (
    feature_scaler.transform(
        history_scaled[features]
    )
)


# ============================================================
# CHECK HISTORY
# ============================================================

sequence_data = history_scaled.tail(
    sequence_length
).copy()

if len(sequence_data) < sequence_length:

    raise ValueError(
        f"Not enough history for {PLOT_TICKER}. "
        f"Required {sequence_length}, "
        f"found {len(sequence_data)}."
    )


print("\n==============================================")
print("MODEL INPUT")
print("==============================================")

print(
    "Input starts:",
    sequence_data["Date"].iloc[0].date()
)

print(
    "Input ends  :",
    sequence_data["Date"].iloc[-1].date()
)

print(
    "Number of input days:",
    len(sequence_data)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained LSTM model...")

model = LSTMModel(
    input_size=len(features),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# GET ACTUAL FUTURE DATA
#
# ONLY used for comparison AFTER predictions.
# It is NEVER used as model input.
# ============================================================

future_actual = ticker_df[
    ticker_df["Date"] > input_date
].copy()

future_actual = future_actual.sort_values(
    "Date"
).reset_index(drop=True)


# ============================================================
# HELPER:
# FIND NEXT TRADING DATE
# ============================================================

future_actual_dates = future_actual["Date"]


# ============================================================
# RECURSIVE PREDICTION
# ============================================================

print("\n==============================================")
print("STARTING 5-DAY RECURSIVE FORECAST")
print("==============================================")


# ------------------------------------------------------------
# We maintain our own predicted history.
#
# IMPORTANT:
# The actual historical data ends at INPUT_DATE.
# After that, predicted prices are used recursively.
# ------------------------------------------------------------

working_history = history.copy()


predictions = []


for step in range(FORECAST_DAYS):

    print("\n")
    print("-" * 70)
    print(f"FORECAST STEP {step + 1}")
    print("-" * 70)


    # ========================================================
    # GET LAST SEQUENCE_LENGTH ROWS
    # ========================================================

    recent = working_history.tail(
        sequence_length
    ).copy()


    if len(recent) < sequence_length:

        raise ValueError(
            "Not enough data for recursive prediction."
        )


    # ========================================================
    # SCALE FEATURES USING TRAINED SCALER
    # ========================================================

    recent_scaled = recent.copy()

    recent_scaled.loc[:, features] = (
        feature_scaler.transform(
            recent_scaled[features]
        )
    )


    # ========================================================
    # CREATE LSTM INPUT
    # ========================================================

    X = recent_scaled[
        features
    ].values.astype(np.float32)


    X = torch.tensor(
        X,
        dtype=torch.float32
    )


    X = X.unsqueeze(0)

    X = X.to(device)


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with torch.no_grad():

        price_output, direction_output = model(X)


    # ========================================================
    # EXTRACT MODEL OUTPUT
    # ========================================================

    pred_log_return = float(
        price_output
        .cpu()
        .numpy()
        .flatten()[0]
    )


    direction_probability = float(
        torch.sigmoid(direction_output)
        .cpu()
        .numpy()
        .flatten()[0]
    )


    # ========================================================
    # CURRENT PRICE
    # ========================================================

    current_close = float(
        working_history["Close"].iloc[-1]
    )


    # ========================================================
    # PREDICT NEXT PRICE
    # ========================================================

    predicted_close = (
        current_close
        * np.exp(pred_log_return)
    )


    # ========================================================
    # PREDICT DIRECTION
    # ========================================================

    predicted_direction = (
        "UP"
        if direction_probability >= 0.5
        else "DOWN"
    )


    # ========================================================
    # DETERMINE NEXT TRADING DATE
    # ========================================================

    last_date = working_history[
        "Date"
    ].iloc[-1]


    future_dates = future_actual_dates[
        future_actual_dates > last_date
    ]


    if len(future_dates) > 0:

        next_date = future_dates[0]

    else:

        # If actual future data is unavailable,
        # use business-day calculation.

        next_date = (
            last_date
            + pd.offsets.BDay(1)
        )


    # ========================================================
    # ACTUAL PRICE
    #
    # Used ONLY for evaluation.
    # ========================================================

    actual_match = future_actual[
        future_actual["Date"] == next_date
    ]


    if len(actual_match) > 0:

        actual_close = float(
            actual_match["Close"].iloc[0]
        )

    else:

        actual_close = np.nan


    # ========================================================
    # ACTUAL DIRECTION
    # ========================================================

    if not np.isnan(actual_close):

        actual_return = (
            actual_close - current_close
        ) / current_close

        actual_direction = (
            "UP"
            if actual_return > 0
            else "DOWN"
        )

        direction_correct = (
            predicted_direction
            == actual_direction
        )

    else:

        actual_return = np.nan
        actual_direction = "N/A"
        direction_correct = False


    # ========================================================
    # STORE RESULT
    # ========================================================

    predictions.append({

        "Date": next_date,

        "Current Close": current_close,

        "Predicted Close": predicted_close,

        "Actual Close": actual_close,

        "Predicted Log Return": pred_log_return,

        "Direction Probability": direction_probability,

        "Predicted Direction": predicted_direction,

        "Actual Direction": actual_direction,

        "Direction Correct": direction_correct,

    })


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        f"Prediction date      : {next_date.date()}"
    )

    print(
        f"Previous close       : {current_close:.4f}"
    )

    print(
        f"Predicted close      : {predicted_close:.4f}"
    )

    print(
        f"Predicted log return : {pred_log_return:.6f}"
    )

    print(
        f"Predicted direction  : {predicted_direction}"
    )

    print(
        f"Direction probability: "
        f"{direction_probability:.4f}"
    )


    if not np.isnan(actual_close):

        error_percent = (
            (predicted_close - actual_close)
            / actual_close
            * 100
        )

        print(
            f"Actual close         : {actual_close:.4f}"
        )

        print(
            f"Actual direction     : "
            f"{actual_direction}"
        )

        print(
            f"Direction correct    : "
            f"{'YES' if direction_correct else 'NO'}"
        )

        print(
            f"Price error          : "
            f"{error_percent:.4f}%"
        )


    # ========================================================
    # ADD PREDICTED ROW TO WORKING HISTORY
    #
    # IMPORTANT:
    # We need to create the next row so that it can be
    # used for the following prediction.
    # ========================================================

    new_row = working_history.iloc[-1].copy()


    new_row["Date"] = next_date

    new_row["Close"] = predicted_close


    # --------------------------------------------------------
    # IMPORTANT:
    # Recalculate quantitative features.
    #
    # We create a temporary raw-price dataframe and run
    # add_quant_features again.
    # --------------------------------------------------------

    temp_history = pd.concat(
        [
            working_history,
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # Recalculate features from predicted price history
    # --------------------------------------------------------

    temp_history = temp_history.drop(
        columns=[
            c for c in [
                "next_date",
                "next_close",
                "next_return",
                "next_log_return",
                "direction_target"
            ]
            if c in temp_history.columns
        ],
        errors="ignore"
    )


    temp_history = add_quant_features(
        temp_history
    )


    # --------------------------------------------------------
    # Keep only the columns needed for next iteration
    # --------------------------------------------------------

    working_history = temp_history.copy()


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    predictions
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n")
print("=" * 100)
print("5-DAY FORECAST RESULTS")
print("=" * 100)

display_columns = [
    "Date",
    "Current Close",
    "Predicted Close",
    "Actual Close",
    "Predicted Direction",
    "Actual Direction",
    "Direction Probability",
    "Direction Correct",
]

print(
    results[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# GRAPH
# ============================================================

print("\n==============================================")
print("CREATING GRAPH")
print("==============================================")


# ------------------------------------------------------------
# Historical starting point
# ------------------------------------------------------------

plot_dates = [
    input_date
] + results["Date"].tolist()


# ------------------------------------------------------------
# Actual prices
#
# Starting price is known.
# Future actual prices are only for comparison.
# ------------------------------------------------------------

actual_prices = [
    float(
        history["Close"].iloc[-1]
    )
]


for value in results["Actual Close"]:

    actual_prices.append(value)


# ------------------------------------------------------------
# Predicted prices
#
# Starting point = actual Feb 18 price
# Then model predictions.
# ------------------------------------------------------------

predicted_prices = [
    float(
        history["Close"].iloc[-1]
    )
] + results[
    "Predicted Close"
].tolist()


plt.figure(
    figsize=(14, 8)
)


# ============================================================
# ACTUAL
# ============================================================

plt.plot(
    plot_dates,
    actual_prices,
    marker="o",
    linewidth=2,
    label="Actual Price"
)


# ============================================================
# PREDICTED
# ============================================================

plt.plot(
    plot_dates,
    predicted_prices,
    marker="o",
    linestyle="--",
    linewidth=2,
    label="Recursive Prediction"
)


# ============================================================
# MARK STARTING POINT
# ============================================================

plt.scatter(
    input_date,
    history["Close"].iloc[-1],
    s=120,
    zorder=5
)


plt.annotate(
    f"Last Actual: "
    f"{history['Close'].iloc[-1]:.2f}",
    (
        input_date,
        history["Close"].iloc[-1]
    ),
    xytext=(-20, 15),
    textcoords="offset points",
    fontsize=10
)


# ============================================================
# LABEL PREDICTIONS
# ============================================================

for _, row in results.iterrows():

    plt.annotate(
        f"{row['Predicted Close']:.2f}",
        (
            row["Date"],
            row["Predicted Close"]
        ),
        xytext=(5, 10),
        textcoords="offset points",
        fontsize=9
    )


# ============================================================
# LABELS
# ============================================================

plt.xlabel(
    "Date"
)

plt.ylabel(
    f"{PLOT_TICKER} Price"
)

plt.title(
    f"{PLOT_TICKER}: 5-Day Recursive Next-Trading-Day Forecast\n"
    f"Last Actual Data: {input_date.date()}"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    rotation=30
)

plt.tight_layout()


# ============================================================
# SAVE GRAPH
# ============================================================

graph_path = "5_day_recursive_prediction.png"

plt.savefig(
    graph_path,
    dpi=150,
    bbox_inches="tight"
)

print(
    f"\nGraph saved as: {graph_path}"
)


# ============================================================
# SHOW GRAPH
# ============================================================

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Ticker              : {PLOT_TICKER}"
)

print(
    f"Last actual date    : "
    f"{input_date.date()}"
)

print(
    f"Forecast days       : "
    f"{FORECAST_DAYS}"
)

print(
    "\nThe model performed recursive "
    "one-day-ahead predictions."
)

print(
    "Future predictions were generated "
    "without using future actual prices."
)