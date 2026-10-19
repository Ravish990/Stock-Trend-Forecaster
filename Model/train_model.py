import copy
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import sys
from Model.model import LSTMModel, AttentionLSTMModel, CrossSectionalScaler
sys.modules['__main__'].CrossSectionalScaler = CrossSectionalScaler

SEED = 42
SEQUENCE = 20
HORIZON = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.25

WEIGHT_DECAY = 1e-4
LEARNING_RATE = 3e-4

# Multi-task loss scales
ALPHA = 20.0  # Scale for SmoothL1 excess return loss
BETA = 1.0    # Scale for excess direction BCE loss

EPOCHS = 15
BATCH_SIZE = 512
PATIENCE = 5

DATA_PATH = "merged_news_target_data.csv"
MODEL_PATH = "best_lstm_stock_model_v4_h5.pth"
SCALER_PATH = "lstm_stock_scalers_V4_h5.pkl"

TRAINING_START_DATE = pd.Timestamp("2021-01-01")
EVALUATION_START_DATE = pd.Timestamp("2025-01-01")
TEST_START_DATE = pd.Timestamp("2025-08-01")
TEST_END_DATE = pd.Timestamp("2026-02-19")

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

STOCK_FEATURES = [f for f in FEATURES if not f.startswith("market_")]
MACRO_FEATURES = [f for f in FEATURES if f.startswith("market_")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class CrossSectionalScaler:
    """
    Standardizes stock features cross-sectionally per date to remove macro collinearity,
    and scales macro features across time.
    """
    def __init__(self, stock_features, macro_features):
        self.stock_features = stock_features
        self.macro_features = macro_features
        self.macro_scaler = StandardScaler()

    def fit(self, df, train_mask):
        train_macro = df.loc[train_mask, self.macro_features]
        self.macro_scaler.fit(train_macro)
        return self

    def transform_df(self, df):
        df = df.copy()
        # Fast vectorized cross-sectional z-score for stock features
        grouped = df.groupby("Date")[self.stock_features]
        means = grouped.transform("mean")
        stds = grouped.transform("std")
        df[self.stock_features] = ((df[self.stock_features] - means) / (stds + 1e-6)).clip(-5.0, 5.0)

        # Macro scaling
        scaled_macro = self.macro_scaler.transform(df[self.macro_features])
        df[self.macro_features] = np.clip(scaled_macro, -5.0, 5.0)
        return df

    def transform(self, array_2d):
        """
        Fallback / inference transform for raw feature matrices (N, len(FEATURES)).
        Stock features are assumed pre-standardized or scaled via standard z-score.
        """
        arr = array_2d.copy()
        macro_indices = [FEATURES.index(f) for f in self.macro_features]
        arr[:, macro_indices] = self.macro_scaler.transform(arr[:, macro_indices])
        return np.clip(arr, -5.0, 5.0)


def build_model():
    return AttentionLSTMModel(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)


def get_data_with_horizon():
    print("Reading dataset...")
    data = pd.read_csv(DATA_PATH)
    required = ["Ticker", "Date", "Close"] + FEATURES
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in {DATA_PATH}: {missing}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Ticker", "Date", "Close"]).copy()
    data["Ticker"] = data["Ticker"].astype(str).str.strip().str.upper()

    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # 5-day horizon targets
    grouped = data.groupby("Ticker", sort=False)
    data["next_close"] = grouped["Close"].shift(-HORIZON)
    data["next_date"] = grouped["Date"].shift(-HORIZON)
    data["next_return"] = (data["next_close"] / data["Close"]) - 1.0
    data["next_log_return"] = np.log(data["next_close"] / data["Close"])

    # Alpha target: 5-day excess return relative to market median
    data["market_median_ret"] = data.groupby("Date")["next_log_return"].transform("median")
    data["excess_log_return"] = data["next_log_return"] - data["market_median_ret"]
    data["excess_direction"] = np.where(
        data["excess_log_return"].notna(),
        (data["excess_log_return"] > 0).astype(float),
        np.nan,
    )
    data["raw_direction"] = np.where(
        data["next_return"].notna(),
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
    data = data.dropna(subset=FEATURES + ["excess_log_return"]).copy()
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    train_mask = (
        (data["Date"] >= TRAINING_START_DATE)
        & (data["Date"] < EVALUATION_START_DATE)
    )

    scaler = CrossSectionalScaler(STOCK_FEATURES, MACRO_FEATURES)
    scaler.fit(data, train_mask)
    data = scaler.transform_df(data)

    return data, scaler


def prepare_sequences(data, split):
    X, y_ret, y_dir = [], [], []
    y_raw_ret, y_raw_dir = [], []
    input_dates, target_dates, tickers = [], [], []

    # Subsample training sequences with stride=2 to eliminate 95% collinearity and 2x training speed
    stride = 2 if split == "train" else 1

    for ticker, group in data.groupby("Ticker", sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        feature_array = group[FEATURES].to_numpy(dtype=np.float32)
        returns = group["excess_log_return"].to_numpy(dtype=np.float32)
        directions = group["excess_direction"].to_numpy(dtype=np.float32)
        raw_rets = group["next_log_return"].to_numpy(dtype=np.float32)
        raw_dirs = group["raw_direction"].to_numpy(dtype=np.float32)
        dates = group["Date"].to_numpy()
        next_dates = group["next_date"].to_numpy()

        for i in range(SEQUENCE - 1, len(group), stride):
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
            y_ret.append(returns[i])
            y_dir.append(directions[i])
            y_raw_ret.append(raw_rets[i])
            y_raw_dir.append(raw_dirs[i])
            input_dates.append(input_date)
            target_dates.append(target_date)
            tickers.append(ticker)

    if not X:
        raise ValueError(f"No sequences generated for split '{split}'.")

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_ret, dtype=np.float32),
        np.asarray(y_dir, dtype=np.float32),
        np.asarray(y_raw_ret, dtype=np.float32),
        np.asarray(y_raw_dir, dtype=np.float32),
        np.asarray(input_dates),
        np.asarray(target_dates),
        np.asarray(tickers),
    )


def prepare_data_for_training():
    t0 = time.time()
    data = get_data_with_horizon()
    print("Preparing cross-sectional features...")
    data, scaler = prepare_features(data)

    print("Extracting train sequences...")
    train_data = prepare_sequences(data, "train")
    print(f"Train sequences: {len(train_data[0]):,} | Alpha UP balance: {(train_data[2]==1).mean():.1%}")

    print("Extracting eval sequences...")
    eval_data = prepare_sequences(data, "evaluation")
    print(f"Eval sequences: {len(eval_data[0]):,} | Alpha UP balance: {(eval_data[2]==1).mean():.1%}")

    print("Extracting test sequences...")
    test_data = prepare_sequences(data, "test")
    print(f"Test sequences: {len(test_data[0]):,} | Alpha UP balance: {(test_data[2]==1).mean():.1%}")
    print(f"Data pipeline ready in {time.time()-t0:.2f}s")

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


def train_model():
    print("=" * 80)
    print("CALIBRATED MULTI-TICKER ATTENTION-LSTM (ALPHA FORECASTING)")
    print("=" * 80)
    print(f"Device: {device} | Features: {len(FEATURES)} | Sequence: {SEQUENCE} | Horizon: {HORIZON}")
    print(f"Hidden: {HIDDEN_SIZE} | Layers: {NUM_LAYERS} | Dropout: {DROPOUT} | Batch: {BATCH_SIZE}")

    train_data, eval_data, test_data, scaler = prepare_data_for_training()

    train_loader = create_dataloader(train_data, shuffle=True)
    eval_loader = create_dataloader(eval_data, shuffle=False)

    model = build_model()
    criterion_ret = nn.SmoothL1Loss(beta=0.005)
    criterion_dir = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
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

        scheduler.step()
        avg_train_loss = train_loss_sum / train_samples

        eval_metrics = run_evaluation(
            model, eval_loader, criterion_ret, criterion_dir,
            desc=f"Epoch [{epoch:02d}/{EPOCHS}] Eval"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Eval Loss: {eval_metrics['loss']:.6f} | "
            f"Alpha RMSE: {eval_metrics['rmse']:.6f} | "
            f"Alpha Acc: {eval_metrics['accuracy']:.2f}% | "
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
    print(f"Alpha accuracy at best  : {best_eval_acc:.2f}%")

    evaluate_saved_model(eval_data, test_data)


def evaluate_saved_model(eval_data=None, test_data=None):
    print("\nEvaluating saved model on held-out test data...")
    scaler = joblib.load(SCALER_PATH)
    loaded_model = build_model()
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    loaded_model.load_state_dict(state)
    loaded_model.eval()

    test_loader = create_dataloader(test_data, shuffle=False)
    criterion_ret = nn.SmoothL1Loss(beta=0.005)
    criterion_dir = nn.BCEWithLogitsLoss()

    test_metrics = run_evaluation(loaded_model, test_loader, criterion_ret, criterion_dir, desc="Final Test")

    # Reconstruct raw return predictions:
    # r_total = market_drift + alpha_z * calibrated_scale
    pred_alphas = test_metrics["pred_return"]
    raw_rets = test_data[3]
    raw_dirs = test_data[4]

    market_drift = 0.0043  # +0.43% 5-day market drift
    # Normalizing alpha predictions to match empirical stock return dispersion
    alpha_std = np.std(pred_alphas) + 1e-6
    calibrated_alphas = (pred_alphas - np.mean(pred_alphas)) / alpha_std * 0.028
    pred_total_rets = market_drift + calibrated_alphas
    pred_raw_dirs = (pred_total_rets > 0).astype(float)

    raw_acc = (pred_raw_dirs == raw_dirs).mean() * 100
    raw_rmse = np.sqrt(((pred_total_rets - raw_rets)**2).mean())
    pred_up_pct = (pred_raw_dirs == 1).mean() * 100
    pred_down_pct = (pred_raw_dirs == 0).mean() * 100
    act_up_pct = (raw_dirs == 1).mean() * 100

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT RAW PRICE FORECAST METRICS (P_t+5 > P_t)")
    print("=" * 70)
    print(f"Total Test Sequences    : {len(pred_raw_dirs):,}")
    print(f"Direction Accuracy      : {raw_acc:.2f}%")
    print(f"Return RMSE             : {raw_rmse:.6f}")
    print(f"Predicted UP %          : {pred_up_pct:.1f}%")
    print(f"Predicted DOWN %        : {pred_down_pct:.1f}%")
    print(f"Actual UP %             : {act_up_pct:.1f}%")


if __name__ == "__main__":
    train_model()