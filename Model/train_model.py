import pickle
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from Data_Processing.target_data import get_data
from Model.model import LSTMModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


HORIZON = 1
SEQUENCE_LENGTH = 60

TRAIN_START = pd.Timestamp("2010-01-01")
VALIDATION_START = pd.Timestamp("2016-01-01")
TEST_START = pd.Timestamp("2018-01-01")
TEST_END = pd.Timestamp("2020-01-01")


BATCH_SIZE = 512
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 12

REGRESSION_TARGET = "next_log_return"
DIRECTION_TARGET = "direction_target"

ALPHA = 0.1
BETA = 1.0

FEATURE_CLIP = 5.0

TARGET_WINSOR_Q = (0.005, 0.995)

MODEL_PATH = f"best_lstm_stock_model_v1_h{HORIZON}.pth"
SCALER_PATH = f"lstm_stock_scalers_v1_h{HORIZON}.pkl"


def create_sequences(
    df,
    features,
    sequence_length=20,
    target_name=REGRESSION_TARGET,
    direction_name=DIRECTION_TARGET,
    min_target_date=None,
    max_target_date=None,
):
    X_sequences = []
    regression_targets = []
    direction_targets = []

    current_closes = []
    next_closes = []
    next_returns = []
    target_dates = []
    tickers = []

    for ticker, group in df.groupby("Ticker", sort=False):
        group = group.sort_values("Date").reset_index(drop=True)

        X = group[features].to_numpy(dtype=np.float32)
        regression = group[target_name].to_numpy(dtype=np.float32)
        direction = group[direction_name].to_numpy(dtype=np.float32)

        current_close = group["Close"].to_numpy(dtype=np.float64)
        next_close = group["next_close"].to_numpy(dtype=np.float64)
        next_return = group["next_return"].to_numpy(dtype=np.float64)
        dates = group["next_date"].to_numpy()

        for i in range(sequence_length - 1, len(group)):
            target_date = dates[i]

            if pd.isna(target_date):
                continue

            if min_target_date is not None and target_date < np.datetime64(min_target_date):
                continue

            if max_target_date is not None and target_date >= np.datetime64(max_target_date):
                continue

            values = X[i - sequence_length + 1:i + 1]
            reg = regression[i]
            direc = direction[i]

            if not np.isfinite(values).all():
                continue

            if not np.isfinite(reg) or not np.isfinite(direc):
                continue
            if not np.isfinite(current_close[i]) or not np.isfinite(next_close[i]):
                continue

            X_sequences.append(values)
            regression_targets.append(reg)
            direction_targets.append(direc)
            current_closes.append(current_close[i])
            next_closes.append(next_close[i])
            next_returns.append(next_return[i])
            target_dates.append(target_date)
            tickers.append(ticker)

    return {
        "X": np.asarray(X_sequences, dtype=np.float32),
        "regression": np.asarray(regression_targets, dtype=np.float32),
        "direction": np.asarray(direction_targets, dtype=np.float32),
        "current_close": np.asarray(current_closes, dtype=np.float64),
        "next_close": np.asarray(next_closes, dtype=np.float64),
        "next_return": np.asarray(next_returns, dtype=np.float64),
        "target_dates": np.asarray(target_dates),
        "tickers": np.asarray(tickers),
    }


def naive_baselines(split):
    """Reference numbers every model result must be compared against."""
    p_up = float(split["direction"].mean())

    return {
        "majority_accuracy": max(p_up, 1.0 - p_up) * 100.0,
        "same_price_rmse": float(np.sqrt(np.mean(
            (split["current_close"] - split["next_close"]) ** 2
        ))),
        "same_price_mae": float(np.mean(
            np.abs(split["current_close"] - split["next_close"])
        )),
    }


def evaluate_model(model, data_loader, target_scaler, device):
    model.eval()

    pred_scaled = []
    actual_scaled = []
    direction_probabilities = []
    direction_actual = []

    with torch.no_grad():
        for X_batch, y_batch, d_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            d_batch = d_batch.to(device)

            return_pred, direction_logit = model(X_batch)
            return_pred = return_pred.squeeze(1)
            direction_logit = direction_logit.squeeze(1)

            probability = torch.sigmoid(direction_logit)

            pred_scaled.append(return_pred.cpu().numpy())
            actual_scaled.append(y_batch.cpu().numpy())
            direction_probabilities.append(probability.cpu().numpy())
            direction_actual.append(d_batch.cpu().numpy())

    pred_scaled = np.concatenate(pred_scaled)
    actual_scaled = np.concatenate(actual_scaled)
    direction_probability = np.concatenate(direction_probabilities)
    direction_actual = np.concatenate(direction_actual)

    pred_log_return = target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()

    actual_log_return = target_scaler.inverse_transform(
        actual_scaled.reshape(-1, 1)
    ).ravel()

    direction_prediction = (direction_probability >= 0.5).astype(np.float32)

    accuracy = float(np.mean(direction_prediction == direction_actual) * 100.0)

    return {
        "pred_scaled": pred_scaled,
        "actual_scaled": actual_scaled,
        "pred_log_return": pred_log_return,
        "actual_log_return": actual_log_return,
        "direction_probability": direction_probability,
        "direction_prediction": direction_prediction,
        "direction_actual": direction_actual,
        "accuracy": accuracy,
        "rmse_log_return": float(
            np.sqrt(np.mean((pred_log_return - actual_log_return) ** 2))
        ),
        "mae_log_return": float(
            np.mean(np.abs(pred_log_return - actual_log_return))
        ),
    }


def price_metrics(split, metrics):
    """Convert predicted log-return into predicted next-day price."""
    predicted_price = split["current_close"] * np.exp(metrics["pred_log_return"])

    actual_price = split["next_close"]
    error = predicted_price - actual_price

    return {
        "predicted_price": predicted_price,
        "actual_price": actual_price,
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }


def print_results(title, split, metrics):
    price = price_metrics(split, metrics)
    naive = naive_baselines(split)

    # Naive return RMSE = error of always predicting a return of zero.
    naive_return_rmse = float(np.sqrt(np.mean(metrics["actual_log_return"] ** 2)))

    print(f"\n===== {title} =====")
    print(f"Log-return RMSE:    {metrics['rmse_log_return']:.6f}"
          f"   (zero-return baseline: {naive_return_rmse:.6f})")
    print(f"Log-return MAE:     {metrics['mae_log_return']:.6f}")
    print(f"Price RMSE:         {price['rmse']:.4f}"
          f"   (same-price baseline: {naive['same_price_rmse']:.4f})")
    print(f"Price MAE:          {price['mae']:.4f}"
          f"   (same-price baseline: {naive['same_price_mae']:.4f})")
    print(f"Directional Acc.:   {metrics['accuracy']:.2f}%"
          f"   (always-majority baseline: {naive['majority_accuracy']:.2f}%)")

    return price, naive


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Prediction horizon: {HORIZON} trading days")
    print(f"Regression target: {REGRESSION_TARGET}")
    print(f"Direction target: {DIRECTION_TARGET}")

    df = get_data(horizon=HORIZON)
    features = [
        # Momentum / trend
        "return_1d",
        "return_5d",
        "return_20d",
        "price_to_MA5",
        "price_to_MA20",
        "volatility_5d",
        "volatility_20d",
        "relative_volume_20",

        # Technical indicators
        "RSI_14",
        "macd_hist",
        "bollinger_pct_b",

        # Market-relative
        "market_return_1d",
        "market_return_5d",
        "excess_return_1d",
        "market_volatility",
        "market_breadth",

        # Beta / residual
        "rolling_beta_20",
        "residual_return_1d",

        # Price structure
        "atr_14",
        "dist_52w_high",
        "dist_52w_low",
    ]

    required = features + [
        "Date",
        "Ticker",
        "Close",
        "next_date",
        "next_close",
        "next_return",
        "next_log_return",
        "direction_target",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    feature_scaler = StandardScaler()
    train_mask = (df["Date"] >= TRAIN_START) & (df["Date"] < VALIDATION_START)

    feature_scaler.fit(df.loc[train_mask, features])
    df.loc[:, features] = feature_scaler.transform(df[features])

    df[features] = df[features].clip(-FEATURE_CLIP, FEATURE_CLIP)

    train_target_mask = (
        (df["next_date"] >= TRAIN_START) & (df["next_date"] < VALIDATION_START)
    )

    lo, hi = df.loc[train_target_mask, REGRESSION_TARGET].quantile(
        list(TARGET_WINSOR_Q)
    )
    print(f"\nTarget winsorization bounds (train only): [{lo:.4f}, {hi:.4f}]")

    df["target_winsorized"] = df[REGRESSION_TARGET].clip(lo, hi)

    target_scaler = StandardScaler()
    target_scaler.fit(df.loc[train_target_mask, ["target_winsorized"]])

    df["regression_target_scaled"] = target_scaler.transform(
        df[["target_winsorized"]]
    ).astype(np.float32)

    train = create_sequences(
        df,
        features,
        sequence_length=SEQUENCE_LENGTH,
        target_name="regression_target_scaled",
        direction_name=DIRECTION_TARGET,
        min_target_date=TRAIN_START,
        max_target_date=VALIDATION_START,
    )

    # Context = the last SEQUENCE_LENGTH rows BEFORE the split starts.
    validation_context = (
        df[df["Date"] < VALIDATION_START]
        .groupby("Ticker")
        .tail(SEQUENCE_LENGTH)
    )

    validation_data = df[
        (df["Date"] >= VALIDATION_START) & (df["Date"] < TEST_START)
    ]

    validation_sequence_df = (
        pd.concat([validation_context, validation_data])
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    val = create_sequences(
        validation_sequence_df,
        features,
        sequence_length=SEQUENCE_LENGTH,
        target_name="regression_target_scaled",
        direction_name=DIRECTION_TARGET,
        min_target_date=VALIDATION_START,
        max_target_date=TEST_START,
    )

    test_context = (
        df[df["Date"] < TEST_START]
        .groupby("Ticker")
        .tail(SEQUENCE_LENGTH)
    )

    test_data = df[(df["Date"] >= TEST_START) & (df["Date"] < TEST_END)]

    test_sequence_df = (
        pd.concat([test_context, test_data])
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    test = create_sequences(
        test_sequence_df,
        features,
        sequence_length=SEQUENCE_LENGTH,
        target_name="regression_target_scaled",
        direction_name=DIRECTION_TARGET,
        min_target_date=TEST_START,
        max_target_date=TEST_END,
    )

    print("\n===== DATA SHAPES =====")
    print("X_train:", train["X"].shape)
    print("X_val:", val["X"].shape)
    print("X_test:", test["X"].shape)
    print("Number of features:", len(features))

    print("\n===== DIRECTION BALANCE =====")
    for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
        p_up = split["direction"].mean() * 100.0
        print(f"{name}: UP={p_up:.2f}% DOWN={100.0 - p_up:.2f}%")

    val_naive = naive_baselines(val)
    test_naive = naive_baselines(test)
    print(
        f"\nBaselines -> Val majority acc: {val_naive['majority_accuracy']:.2f}% | "
        f"Test majority acc: {test_naive['majority_accuracy']:.2f}%"
    )
    print(
        f"Baselines -> Val same-price RMSE: {val_naive['same_price_rmse']:.4f} | "
        f"Test same-price RMSE: {test_naive['same_price_rmse']:.4f}"
    )

    # --------------------------------------------------------
    # PyTorch datasets
    # --------------------------------------------------------
    X_train = torch.tensor(train["X"], dtype=torch.float32)
    y_train = torch.tensor(train["regression"], dtype=torch.float32)
    d_train = torch.tensor(train["direction"], dtype=torch.float32)

    X_val = torch.tensor(val["X"], dtype=torch.float32)
    y_val = torch.tensor(val["regression"], dtype=torch.float32)
    d_val = torch.tensor(val["direction"], dtype=torch.float32)

    X_test = torch.tensor(test["X"], dtype=torch.float32)
    y_test = torch.tensor(test["regression"], dtype=torch.float32)
    d_test = torch.tensor(test["direction"], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train, d_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val, d_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test, d_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = LSTMModel(
        input_size=len(features),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    regression_loss_fn = torch.nn.HuberLoss(delta=1.0)

    positives = d_train.sum().item()
    negatives = len(d_train) - positives
    pos_weight_value = negatives / positives if positives > 0 else 1.0

    direction_loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight_value, dtype=torch.float32, device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    patience_counter = 0
    best_accuracy = 0.0
    best_log_return_rmse = float("inf")
    saved_any = False

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch, d_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            d_batch = d_batch.to(device)

            optimizer.zero_grad()

            return_pred, direction_logit = model(X_batch)
            return_pred = return_pred.squeeze(1)
            direction_logit = direction_logit.squeeze(1)

            regression_loss = regression_loss_fn(return_pred, y_batch)
            direction_loss = direction_loss_fn(direction_logit, d_batch)

            loss = ALPHA * regression_loss + BETA * direction_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / max(1, len(train_loader))

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch, d_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                d_batch = d_batch.to(device)

                return_pred, direction_logit = model(X_batch)
                return_pred = return_pred.squeeze(1)
                direction_logit = direction_logit.squeeze(1)

                regression_loss = regression_loss_fn(return_pred, y_batch)
                direction_loss = direction_loss_fn(direction_logit, d_batch)

                total_val_loss += (
                    ALPHA * regression_loss + BETA * direction_loss
                ).item()

        val_loss = total_val_loss / max(1, len(val_loader))
        scheduler.step(val_loss)

        metrics = evaluate_model(model, val_loader, target_scaler, device)
        price = price_metrics(val, metrics)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"TrainLoss={train_loss:.5f} "
            f"ValLoss={val_loss:.5f} "
            f"ReturnRMSE={metrics['rmse_log_return']:.6f} "
            f"PriceRMSE={price['rmse']:.4f} "
            f"Acc={metrics['accuracy']:.2f}% "
            f"LR={current_lr:.6f}"
        )

        directional_accuracy_improved = metrics["accuracy"] > best_accuracy
        log_return_rmse_improved = metrics["rmse_log_return"] < best_log_return_rmse

        if directional_accuracy_improved:
            best_accuracy = metrics["accuracy"]
            best_log_return_rmse = metrics["rmse_log_return"]

            torch.save(model.state_dict(), MODEL_PATH)
            saved_any = True
            print(f"  Saved best model to {MODEL_PATH}")

            patience_counter = 0
        else:
            best_log_return_rmse = min(best_log_return_rmse, metrics["rmse_log_return"])

            patience_counter += 1
            print(f"  No improvement. Patience counter: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # Safety: if the strict rule never saved, save the final model so loading works.
    if not saved_any:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"\nNo epoch met the save rule; saved last model to {MODEL_PATH}")

    print("\nLoading best model...")
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    val_metrics = evaluate_model(model, val_loader, target_scaler, device)
    test_metrics = evaluate_model(model, test_loader, target_scaler, device)

    print_results("VALIDATION RESULTS", val, val_metrics)
    print_results("TEST RESULTS", test, test_metrics)

    scaler_bundle = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": features,
        "sequence_length": SEQUENCE_LENGTH,
        "regression_target": REGRESSION_TARGET,
        "direction_target": DIRECTION_TARGET,
        "feature_clip": FEATURE_CLIP,
        "target_winsor_bounds": (float(lo), float(hi)),
    }

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler_bundle, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scalers saved to: {SCALER_PATH}")

    return model


if __name__ == "__main__":
    train_model()