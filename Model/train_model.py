import copy
import math
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Model.model import LSTMModel

SEED = 42

SEQUENCE = 10
HORIZON = 5
HIDDEN_SIZE = 32       
NUM_LAYERS = 2
DROPOUT = 0.5         

WEIGHT_DECAY = 0.005  
LEARNING_RATE = 0.0002

ALPHA = 0.5
BETA = 1.0

EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 5

DATA_PATH = "merged_news_target_data.csv"

MODEL_PATH = "best_lstm_stock_model_V2_h5.pth"
SCALER_PATH = "lstm_stock_scalers_V2_h5.pkl"

TRAINING_START_DATE = pd.Timestamp("2020-01-01")
EVALUATION_START_DATE = pd.Timestamp("2024-01-01")
TEST_START_DATE = pd.Timestamp("2025-01-01")
TEST_END_DATE = pd.Timestamp("2026-02-01")

FEATURES = [
    "return_1d", "return_5d", "return_20d", "price_to_MA5", "price_to_MA20",
    "volatility_5d", "volatility_20d", "relative_volume_20", "RSI_14",
    "macd_hist", "bollinger_pct_b", "market_return_1d", "market_return_5d",
    "excess_return_1d", "market_volatility", "market_breadth",
    "rolling_beta_20", "residual_return_1d", "atr_14", "dist_52w_high",
    "dist_52w_low", "News_Count", "Avg_Sentiment", "Avg_Relevance",
    "Sentiment_Std", "Max_Sentiment", "Min_Sentiment", "Weighted_Sentiment",
    "Positive_Count", "Negative_Count", "Neutral_Count",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("MULTI-TICKER LSTM STOCK FORECASTER (ANTI-OVERFITTING)")
print("=" * 70)
print(f"Device: {device}")
print(f"Features: {len(FEATURES)}")
print(f"Sequence length: {SEQUENCE}")
print(f"Prediction horizon: {HORIZON} observations")
print(f"Batch size: {BATCH_SIZE}")


def build_model():
    return LSTMModel(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)


def get_data_with_horizon():
    data = pd.read_csv(DATA_PATH)
    required = ["Ticker", "Date", "Close"] + FEATURES
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in {DATA_PATH}: {missing}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Ticker", "Date", "Close"]).copy()
    data["Ticker"] = data["Ticker"].astype(str)

    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    duplicates = data.duplicated(["Ticker", "Date"], keep=False)

    if duplicates.any():
        examples = data.loc[duplicates, ["Ticker", "Date"]].head(10)
        raise ValueError(
            "Duplicate Ticker/Date rows detected. Resolve them before "
            f"training.\nExamples:\n{examples.to_string(index=False)}"
        )

    grouped = data.groupby("Ticker", sort=False)
    data["next_close"] = grouped["Close"].shift(-HORIZON)
    data["next_date"] = grouped["Date"].shift(-HORIZON)
    data["next_return"] = (data["next_close"] / data["Close"]) - 1.0
    data["next_log_return"] = np.log(data["next_close"] / data["Close"])
    data["direction_target"] = np.where(
        data["next_close"].notna(),
        (data["next_return"] > 0).astype(float),
        np.nan,
    )
    return data


def prepare_features(data):
    data = data.copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    data[FEATURES] = data.groupby("Ticker", sort=False)[FEATURES].ffill()
    data = data.dropna(subset=FEATURES).copy()
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    scaler_fit_mask = (
        (data["Date"] >= TRAINING_START_DATE)
        & (data["Date"] < EVALUATION_START_DATE)
    )
    scaler_fit_data = data.loc[scaler_fit_mask, FEATURES]

    if len(scaler_fit_data) == 0:
        raise ValueError("No training-period rows available to fit StandardScaler.")

    scaler = StandardScaler()
    scaler.fit(scaler_fit_data)
    scaled_features = scaler.transform(data[FEATURES])
    data.loc[:, FEATURES] = scaled_features.astype(np.float32)

    if not np.isfinite(data[FEATURES].to_numpy()).all():
        raise ValueError("Non-finite values remain after feature scaling.")

    return data, scaler


def prepare_sequences(data, split):
    X, y_return, y_direction = [], [], []
    input_dates, target_dates, tickers = [], [], []

    for ticker, group in data.groupby("Ticker", sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        feature_array = group[FEATURES].to_numpy(dtype=np.float32)
        returns = group["next_log_return"].to_numpy(dtype=np.float32)
        directions = group["direction_target"].to_numpy(dtype=np.float32)
        dates = group["Date"].to_numpy()
        next_dates = group["next_date"].to_numpy()

        for i in range(SEQUENCE - 1, len(group)):
            if (
                not np.isfinite(returns[i])
                or not np.isfinite(directions[i])
                or pd.isna(next_dates[i])
            ):
                continue

            input_date = pd.Timestamp(dates[i])
            target_date = pd.Timestamp(next_dates[i])

            if split == "train":
                keep = (
                    input_date >= TRAINING_START_DATE
                    and input_date < EVALUATION_START_DATE
                    and target_date < EVALUATION_START_DATE
                )
            elif split == "evaluation":
                keep = (
                    target_date >= EVALUATION_START_DATE
                    and target_date < TEST_START_DATE
                )
            elif split == "test":
                keep = (
                    target_date >= TEST_START_DATE
                    and target_date <= TEST_END_DATE
                )
            else:
                raise ValueError(f"Unknown split: {split}")

            if not keep:
                continue

            window = feature_array[i - SEQUENCE + 1 : i + 1]
            if window.shape != (SEQUENCE, len(FEATURES)):
                continue

            X.append(window)
            y_return.append(returns[i])
            y_direction.append(directions[i])
            input_dates.append(input_date)
            target_dates.append(target_date)
            tickers.append(ticker)

    if not X:
        raise ValueError(f"No sequences generated for split '{split}'.")

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_return, dtype=np.float32),
        np.asarray(y_direction, dtype=np.float32),
        np.asarray(input_dates),
        np.asarray(target_dates),
        np.asarray(tickers),
    )


def prepare_data_for_training():
    print("\nLoading dataset...")
    data = get_data_with_horizon()
    
    print("\nCleaning features and fitting StandardScaler...")
    data, scaler = prepare_features(data)

    train_data = prepare_sequences(data, "train")
    eval_data = prepare_sequences(data, "evaluation")
    test_data = prepare_sequences(data, "test")

    return train_data, eval_data, test_data, scaler


def create_dataloader(split_data, shuffle=False):
    X, y_ret, y_dir = split_data[:3]
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y_ret).float(),
        torch.from_numpy(y_dir).float(),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def run_evaluation(model, loader, criterion_ret, criterion_dir, desc="Evaluating"):
    model.eval()
    total_loss, total_samples = 0.0, 0
    sum_squared_errors, correct_directions, positive_labels = 0.0, 0, 0
    all_predictions, all_returns, all_logits, all_directions = [], [], [], []

    with torch.no_grad():
        # --- ADDED TQDM HERE ---
        for X_batch, y_ret, y_dir in tqdm(loader, desc=desc, leave=False):
            X_batch = X_batch.to(device)
            y_ret = y_ret.to(device)
            y_dir = y_dir.to(device)

            pred_ret, pred_dir = model(X_batch)
            pred_ret = pred_ret.reshape(-1)
            pred_dir = pred_dir.reshape(-1)

            loss_ret = criterion_ret(pred_ret, y_ret)
            loss_dir = criterion_dir(pred_dir, y_dir)
            loss = ALPHA * loss_ret + BETA * loss_dir

            batch_size = y_dir.numel()
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            sum_squared_errors += (pred_ret - y_ret).square().sum().item()
            pred_class = (pred_dir >= 0).float()
            correct_directions += (pred_class == y_dir).sum().item()
            positive_labels += y_dir.sum().item()

            all_predictions.append(pred_ret.cpu().numpy())
            all_returns.append(y_ret.cpu().numpy())
            all_logits.append(pred_dir.cpu().numpy())
            all_directions.append(y_dir.cpu().numpy())

    rmse = math.sqrt(sum_squared_errors / total_samples)
    accuracy = 100.0 * correct_directions / total_samples

    return {
        "loss": total_loss / total_samples,
        "rmse": rmse,
        "accuracy": accuracy,
        "up_fraction": positive_labels / total_samples,
        "pred_return": np.concatenate(all_predictions),
        "actual_return": np.concatenate(all_returns),
        "pred_logits": np.concatenate(all_logits),
        "actual_direction": np.concatenate(all_directions),
        "samples": total_samples,
    }


def print_metrics(name, metrics):
    majority_baseline = 100.0 * max(metrics["up_fraction"], 1.0 - metrics["up_fraction"])
    print(f"\n{name} METRICS")
    print("-" * 50)
    print(f"Samples                  : {metrics['samples']:,}")
    print(f"Combined loss            : {metrics['loss']:.6f}")
    print(f"Log-return RMSE          : {metrics['rmse']:.6f}")
    print(f"Direction accuracy       : {metrics['accuracy']:.2f}%")
    print(f"Majority-class baseline  : {majority_baseline:.2f}%")
    print(f"Actual UP fraction       : {100 * metrics['up_fraction']:.2f}%")


def train_model():
    train_data, eval_data, test_data, scaler = prepare_data_for_training()

    train_loader = create_dataloader(train_data, shuffle=True)
    eval_loader = create_dataloader(eval_data, shuffle=False)

    model = build_model()
    criterion_ret = nn.MSELoss()
    criterion_dir = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )

    best_eval_loss = float("inf")
    best_eval_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting training...")
    print("=" * 90)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        # --- ADDED TQDM HERE ---
        train_iterator = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS}] Train", leave=False)

        for X_batch, y_ret, y_dir in train_iterator:
            X_batch = X_batch.to(device)
            y_ret = y_ret.to(device)
            y_dir = y_dir.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_ret, pred_dir = model(X_batch)
            pred_ret = pred_ret.reshape(-1)
            pred_dir = pred_dir.reshape(-1)

            loss_ret = criterion_ret(pred_ret, y_ret)
            loss_dir = criterion_dir(pred_dir, y_dir)
            loss = ALPHA * loss_ret + BETA * loss_dir

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n = y_dir.numel()
            train_loss_sum += loss.item() * n
            train_samples += n

        avg_train_loss = train_loss_sum / train_samples

        eval_metrics = run_evaluation(
            model, eval_loader, criterion_ret, criterion_dir, 
            desc=f"Epoch [{epoch:02d}/{EPOCHS}] Eval"
        )

        scheduler.step(eval_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Eval Loss: {eval_metrics['loss']:.6f} | "
            f"Log-return RMSE: {eval_metrics['rmse']:.6f} | "
            f"Direction Acc: {eval_metrics['accuracy']:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        if eval_metrics["loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            best_eval_acc = eval_metrics["accuracy"]
            best_epoch = epoch
            patience_counter = 0

            best_state = copy.deepcopy({
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            })

            torch.save(best_state, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)

            print("  --> Best validation loss improved. Model and scaler saved.")
        else:
            patience_counter += 1
            print(f"  --> No validation-loss improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch              : {best_epoch}")
    print(f"Best validation loss    : {best_eval_loss:.6f}")
    print(f"Accuracy at best loss   : {best_eval_acc:.2f}%")

    evaluate_saved_model(eval_data, test_data)


def evaluate_saved_model(eval_data=None, test_data=None):
    print("\nLoading saved model and scaler...")
    scaler = joblib.load(SCALER_PATH)
    loaded_model = build_model()
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    loaded_model.load_state_dict(state)
    loaded_model.eval()

    if eval_data is None or test_data is None:
        data = get_data_with_horizon()
        data, _ = prepare_features(data)
        eval_data = prepare_sequences(data, "evaluation")
        test_data = prepare_sequences(data, "test")

    eval_loader = create_dataloader(eval_data, shuffle=False)
    test_loader = create_dataloader(test_data, shuffle=False)
    criterion_ret = nn.MSELoss()
    criterion_dir = nn.BCEWithLogitsLoss()

    eval_metrics = run_evaluation(loaded_model, eval_loader, criterion_ret, criterion_dir, desc="Final Eval")
    print_metrics("SAVED MODEL / EVALUATION", eval_metrics)

    test_metrics = run_evaluation(loaded_model, test_loader, criterion_ret, criterion_dir, desc="Final Test")
    print_metrics("FINAL HELD-OUT TEST", test_metrics)


if __name__ == "__main__":
    train_model()