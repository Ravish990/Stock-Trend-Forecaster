import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Model.model import LSTMModel

MODEL_PATH = Path("Model/best_lstm_stock_model_v1_h5.pth")
SCALER_PATH = Path("Model/lstm_stock_scalers_v1_h5.pkl")

PLOT_TICKER = "AAPL"
INPUT_DATE = "2026-02-18"

HORIZON = 5
SEQUENCE_LENGTH = 60

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD SCALERS AND TRAINING CONFIGURATION
# ============================================================

def load_scalers():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}\n"
            "Check that training saved the scaler bundle."
        )

    with open(SCALER_PATH, "rb") as f:
        bundle = pickle.load(f)

    required_keys = [
        "feature_scaler",
        "target_scaler",
        "features",
        "sequence_length",
    ]

    missing = [key for key in required_keys if key not in bundle]

    if missing:
        raise KeyError(
            f"Scaler bundle is missing keys: {missing}"
        )

    features = bundle["features"]
    sequence_length = int(bundle["sequence_length"])
    horizon = int(bundle.get("horizon", HORIZON))
    feature_clip = float(bundle.get("feature_clip", 5.0))

    if horizon != HORIZON:
        raise ValueError(
            f"Model bundle horizon is {horizon}, "
            f"but load_model.py expects {HORIZON}."
        )

    return (
        bundle,
        features,
        sequence_length,
        horizon,
        feature_clip,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_trained_model(input_size):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = LSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support either a plain state_dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# LOAD DATA
# ============================================================

def load_data(features):
    df = pd.read_csv("merged_news_target_data.csv")

    required_columns = ["Ticker", "Date", "Close"] + features
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Your dataset is missing required columns:\n"
            f"{missing_columns}\n\n"
            "Ensure that the dataset contains all features "
            "used during training."
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["Close"] = pd.to_numeric(
        df["Close"], errors="coerce"
    )

    df = df.dropna(
        subset=["Ticker", "Date", "Close"]
    )

    df = df[df["Close"] > 0]

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    return df


# ============================================================
# FIND THE INPUT SEQUENCE
# ============================================================

def prepare_input(
    df,
    features,
    feature_scaler,
    sequence_length,
    feature_clip,
):
    ticker_df = df[
        df["Ticker"].astype(str).str.upper()
        == PLOT_TICKER.upper()
    ].copy()

    if ticker_df.empty:
        raise ValueError(
            f"No data found for ticker {PLOT_TICKER}."
        )

    requested_date = pd.Timestamp(INPUT_DATE)

    # Use the requested date if available; otherwise use
    # the latest available trading observation before it.
    available = ticker_df[
        ticker_df["Date"] <= requested_date
    ].copy()

    if available.empty:
        raise ValueError(
            f"No {PLOT_TICKER} observations available "
            f"on or before {INPUT_DATE}."
        )

    current_row = available.iloc[-1]
    current_date = current_row["Date"]
    current_close = float(current_row["Close"])

    # The input window must contain valid feature values.
    valid_rows = available.dropna(
        subset=features
    ).copy()

    valid_rows = valid_rows[
        np.isfinite(
            valid_rows[features].to_numpy(dtype=np.float64)
        ).all(axis=1)
    ]

    # Ensure the selected prediction date itself has valid features.
    valid_current = valid_rows[
        valid_rows["Date"] == current_date
    ]

    if valid_current.empty:
        raise ValueError(
            f"The latest selected observation ({current_date.date()}) "
            "contains missing or non-finite features. "
            "Inspect feature engineering in Data_Processing/news_data.py and Data_Processing/target_data.py."
        )

    # Use the final 60 valid historical feature rows.
    window_df = valid_rows.tail(sequence_length)

    if len(window_df) < sequence_length:
        raise ValueError(
            f"Need {sequence_length} valid observations for "
            f"{PLOT_TICKER}, but only found {len(window_df)}."
        )

    raw_features = window_df[features].to_numpy(
        dtype=np.float32
    )

    scaled_features = feature_scaler.transform(raw_features)

    # Match the feature clipping used during training.
    scaled_features = np.clip(
        scaled_features,
        -feature_clip,
        feature_clip,
    )

    X = torch.tensor(
        scaled_features,
        dtype=torch.float32,
    ).unsqueeze(0).to(DEVICE)

    return X, current_date, current_close, ticker_df


# ============================================================
# PREDICT FIVE-TRADING-DAY-AHEAD PRICE
# ============================================================

def predict(model, X, target_scaler, current_close):
    with torch.no_grad():
        outputs = model(X)

    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise ValueError(
            "Expected LSTMModel.forward() to return "
            "(return_prediction, direction_logit). "
            "Check Model/model.py."
        )

    return_output = outputs[0]
    direction_output = outputs[1]

    # The regression output is in standardized target space.
    scaled_return = (
        return_output.detach()
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )

    predicted_log_return = float(
        target_scaler.inverse_transform(
            scaled_return
        )[0, 0]
    )

    # Convert the predicted log return to a price.
    predicted_close = current_close * np.exp(
        predicted_log_return
    )

    # Direction output is a logit, so apply sigmoid.
    direction_logit = float(
        direction_output.detach()
        .cpu()
        .numpy()
        .reshape(-1)[0]
    )

    direction_probability = float(
        1.0 / (1.0 + np.exp(-np.clip(
            direction_logit, -50, 50
        )))
    )

    predicted_direction = (
        "UP" if direction_probability >= 0.5 else "DOWN"
    )

    return (
        predicted_log_return,
        predicted_close,
        direction_probability,
        predicted_direction,
    )


# ============================================================
# OPTIONAL: FIND ACTUAL CLOSE FIVE OBSERVATIONS LATER
# ============================================================

def get_actual_future_close(ticker_df, current_date, horizon):
    """
    This is for retrospective evaluation only.
    It must not be used as an input to the prediction.
    """
    future_rows = ticker_df[
        ticker_df["Date"] > current_date
    ].sort_values("Date")

    if len(future_rows) < horizon:
        return None, None

    target_row = future_rows.iloc[horizon - 1]

    return (
        float(target_row["Close"]),
        target_row["Date"],
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FIVE-TRADING-DAY-AHEAD STOCK FORECAST")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Ticker: {PLOT_TICKER}")
    print(f"Requested input date: {INPUT_DATE}")

    (
        bundle,
        features,
        sequence_length,
        horizon,
        feature_clip,
    ) = load_scalers()

    print(f"Horizon: {horizon} trading observations")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of features: {len(features)}")

    print("\nLoading dataset...")
    df = load_data(features)

    print(f"Total dataset rows: {len(df):,}")

    feature_scaler = bundle["feature_scaler"]
    target_scaler = bundle["target_scaler"]

    # Keep the architecture identical to training.
    model = load_trained_model(
        input_size=len(features)
    )

    X, current_date, current_close, ticker_df = prepare_input(
        df=df,
        features=features,
        feature_scaler=feature_scaler,
        sequence_length=sequence_length,
        feature_clip=feature_clip,
    )

    print(f"\nInput observation date: {current_date.date()}")
    print(f"Current close: ${current_close:.4f}")
    print(f"Input tensor shape: {tuple(X.shape)}")

    (
        predicted_log_return,
        predicted_close,
        direction_probability,
        predicted_direction,
    ) = predict(
        model=model,
        X=X,
        target_scaler=target_scaler,
        current_close=current_close,
    )

    actual_close, actual_date = get_actual_future_close(
        ticker_df,
        current_date,
        horizon,
    )

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)

    print(f"Ticker: {PLOT_TICKER}")
    print(f"Input date: {current_date.date()}")
    print(f"Input close: ${current_close:.4f}")
    print(f"Forecast horizon: {horizon} trading observations")
    print(f"Predicted log return: {predicted_log_return:.6f}")
    print(
        f"Predicted percentage return: "
        f"{100 * np.expm1(predicted_log_return):.3f}%"
    )
    print(f"Predicted close: ${predicted_close:.4f}")
    print(f"Predicted direction: {predicted_direction}")
    print(
        f"Model UP probability: "
        f"{100 * direction_probability:.2f}%"
    )

    if actual_close is not None:
        actual_return = actual_close / current_close - 1.0
        actual_direction = (
            "UP" if actual_close > current_close else "DOWN"
        )

        print("\nRetrospective comparison:")
        print(f"Actual target date: {actual_date.date()}")
        print(f"Actual target close: ${actual_close:.4f}")
        print(
            f"Actual percentage return: "
            f"{100 * actual_return:.3f}%"
        )
        print(f"Actual direction: {actual_direction}")
        print(
            f"Absolute price error: "
            f"${abs(predicted_close - actual_close):.4f}"
        )
        print(
            "Direction correct: "
            f"{predicted_direction == actual_direction}"
        )
    else:
        print(
            "\nActual future close unavailable: "
            "the dataset does not contain enough future observations."
        )

    print("\nNote: This is a model forecast, not financial advice.")


if __name__ == "__main__":
    main()
