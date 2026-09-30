import random
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from Data_Processing.target_data import get_data
from Model.model import LSTMModel


# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Configuration
# ============================================================
HORIZON = 1                    # next trading day
SEQUENCE_LENGTH = 20

VALIDATION_START = pd.Timestamp("2012-01-01")
TEST_START = pd.Timestamp("2015-01-01")

BETA_WINDOWS = (20, 60)
VOL_WINDOW = 20

BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 7

# Project targets:
# 1) regression: next-day log return
# 2) classification: next-day raw UP/DOWN
REGRESSION_TARGET = "next_log_return"
DIRECTION_TARGET = "direction_target"

ALPHA = 1.0
BETA = 1.0

MODEL_PATH = "best_lstm_stock_model_v5.pth"
SCALER_PATH = "lstm_stock_scalers_v5.pkl"


# ============================================================
# Quant feature engineering
# ============================================================
def add_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stationary/relative market features using data known at t.

    get_data() already builds several market features
    (market_return_1d, market_return_5d, market_volatility,
    market_breadth). Those are KEPT as-is. This function only adds
    columns that get_data() does not already provide, so nothing is
    silently duplicated or renamed by the merge.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if "return_1d" not in df.columns:
        df["return_1d"] = df.groupby("Ticker")["Close"].pct_change()

    if "return_5d" not in df.columns:
        df["return_5d"] = df.groupby("Ticker")["Close"].pct_change(5)

    if "volatility_5d" not in df.columns:
        df["volatility_5d"] = (
            df.groupby("Ticker")["return_1d"]
            .transform(lambda x: x.rolling(5).std())
        )

    if "volatility_20d" not in df.columns:
        df["volatility_20d"] = (
            df.groupby("Ticker")["return_1d"]
            .transform(lambda x: x.rolling(VOL_WINDOW).std())
        )

    # --------------------------------------------------------
    # Equal-weighted market proxy (date-level features).
    # Only merge columns that are not already in the dataframe.
    # --------------------------------------------------------
    market_r = df.groupby("Date")["return_1d"].mean().sort_index()

    market_df = pd.DataFrame({"market_return_1d": market_r})

    market_df["market_return_5d"] = (
        (1.0 + market_r).rolling(5).apply(np.prod, raw=True) - 1.0
    )

    market_df["market_trend_20d"] = (
        (1.0 + market_r).rolling(20).apply(np.prod, raw=True) - 1.0
    )

    market_df = market_df.reset_index()

    new_cols = [
        c for c in market_df.columns
        if c == "Date" or c not in df.columns
    ]
    df = df.merge(market_df[new_cols], on="Date", how="left")

    if "excess_return_1d" not in df.columns:
        df["excess_return_1d"] = df["return_1d"] - df["market_return_1d"]

    # --------------------------------------------------------
    # Rolling beta + cumulative residual return.
    # (Overwrites the get_data() beta with the same definition and
    # adds the cumulative residual columns used as features.)
    # --------------------------------------------------------
    for window in BETA_WINDOWS:
        beta_col = f"rolling_beta_{window}"
        residual_col = f"residual_return_{window}d"

        df[beta_col] = np.nan
        df[residual_col] = np.nan

        for ticker, idx in df.groupby("Ticker", sort=False).groups.items():
            idx = np.asarray(idx)

            stock_r = df.loc[idx, "return_1d"]
            mkt_r = df.loc[idx, "market_return_1d"]

            cov = stock_r.rolling(window).cov(mkt_r)
            var = mkt_r.rolling(window).var()

            beta = cov / var.replace(0.0, np.nan)
            residual = stock_r - beta * mkt_r

            cumulative_residual = (
                (1.0 + residual)
                .rolling(window)
                .apply(np.prod, raw=True)
                - 1.0
            )

            df.loc[idx, beta_col] = beta.to_numpy()
            df.loc[idx, residual_col] = cumulative_residual.to_numpy()

    # --------------------------------------------------------
    # Relative volume
    # --------------------------------------------------------
    volume_col = None
    if "Volume" in df.columns:
        volume_col = "Volume"
    elif "volume" in df.columns:
        volume_col = "volume"

    if "volume_change" not in df.columns:
        if volume_col is not None:
            df["volume_change"] = (
                df.groupby("Ticker")[volume_col].pct_change()
            )
        else:
            df["volume_change"] = 0.0

    if volume_col is not None:
        avg_volume = (
            df.groupby("Ticker")[volume_col]
            .transform(lambda x: x.rolling(20).mean())
        )
        df["relative_volume_20d"] = df[volume_col] / avg_volume - 1.0
    else:
        df["relative_volume_20d"] = df["volume_change"]

    return df


# ============================================================
# Targets
# ============================================================
def add_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Regression target: next_log_return = log(Close[t+h] / Close[t])
    Direction target : 1 if Close[t+h] > Close[t], else 0
    """
    df = df.copy()

    df["next_close"] = df.groupby("Ticker")["Close"].shift(-horizon)
    df["next_date"] = df.groupby("Ticker")["Date"].shift(-horizon)

    df["next_return"] = (df["next_close"] - df["Close"]) / df["Close"]
    df["next_log_return"] = np.log(df["next_close"] / df["Close"])

    df["direction_target"] = (df["next_return"] > 0).astype(np.float32)

    return df


# ============================================================
# Sequence creation
# ============================================================
def create_sequences(
    df: pd.DataFrame,
    features,
    sequence_length=20,
    target_name=REGRESSION_TARGET,
    direction_name=DIRECTION_TARGET,
    min_target_date=None,
    max_target_date=None,
):
    """
    For row i (day t), the input window is the sequence_length days
    ENDING AT day i (rows i-L+1 ... i), and the targets are the move
    from close[i] to close[i+h]. The most recent day is therefore
    inside the input window.
    """
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

            if min_target_date is not None:
                if target_date < np.datetime64(min_target_date):
                    continue

            if max_target_date is not None:
                if target_date >= np.datetime64(max_target_date):
                    continue

            values = X[i - sequence_length + 1: i + 1]
            reg = regression[i]
            direc = direction[i]

            if not np.isfinite(values).all():
                continue
            if not np.isfinite(reg) or not np.isfinite(next_return[i]):
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


# ============================================================
# Metrics
# ============================================================
def safe_ic(pred, actual):
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    valid = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[valid]
    actual = actual[valid]

    if len(pred) < 2 or np.std(pred) == 0 or np.std(actual) == 0:
        return np.nan

    return float(np.corrcoef(pred, actual)[0, 1])


def safe_spearman_ic(pred, actual):
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    valid = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[valid]
    actual = actual[valid]

    if len(pred) < 2:
        return np.nan

    pred_rank = pd.Series(pred).rank(method="average").to_numpy()
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    return safe_ic(pred_rank, actual_rank)


def per_date_rank_ic(split, metrics, min_stocks=20):
    """Cross-sectional rank IC, computed separately for each date and
    then averaged.

    A pooled IC mixes two things: market-timing (all stocks rise
    together on a strong day) and stock-selection (which stocks rise
    more than others on that day). Computing IC within each date
    removes the market-level component, so this number measures
    stock-selection skill only.

    Returns mean IC, IC std, ICIR (mean/std) and the fraction of
    dates with positive IC.
    """
    frame = pd.DataFrame({
        "Date": pd.to_datetime(split["target_dates"]),
        "Pred": metrics["pred_log_return"],
        "Actual": split["next_return"],
    })

    counts = frame.groupby("Date")["Pred"].transform("size")
    frame = frame[counts >= min_stocks]

    if frame.empty:
        return {"mean_ic": np.nan, "ic_std": np.nan,
                "icir": np.nan, "pct_positive": np.nan, "days": 0}

    frame["p_rank"] = frame.groupby("Date")["Pred"].rank()
    frame["a_rank"] = frame.groupby("Date")["Actual"].rank()

    daily_ic = frame.groupby("Date").apply(
        lambda g: g["p_rank"].corr(g["a_rank"])
    ).dropna()

    if len(daily_ic) < 2:
        return {"mean_ic": np.nan, "ic_std": np.nan,
                "icir": np.nan, "pct_positive": np.nan,
                "days": len(daily_ic)}

    mean_ic = float(daily_ic.mean())
    ic_std = float(daily_ic.std(ddof=1))

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": mean_ic / ic_std if ic_std > 0 else np.nan,
        "pct_positive": float((daily_ic > 0).mean() * 100.0),
        "days": int(len(daily_ic)),
    }


def naive_baselines(split):
    """Reference numbers every model result must be compared against."""
    p_up = float(split["direction"].mean())

    return {
        # Always predict the more common class
        "majority_accuracy": max(p_up, 1.0 - p_up) * 100.0,
        # Predict tomorrow's price == today's price
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

    balanced_accuracy = float(
        balanced_accuracy_score(direction_actual, direction_prediction) * 100.0
    )

    try:
        auc = float(roc_auc_score(direction_actual, direction_probability))
    except ValueError:
        auc = np.nan

    return {
        "pred_scaled": pred_scaled,
        "actual_scaled": actual_scaled,
        "pred_log_return": pred_log_return,
        "actual_log_return": actual_log_return,
        "direction_probability": direction_probability,
        "direction_prediction": direction_prediction,
        "direction_actual": direction_actual,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
        "pearson_ic": safe_ic(pred_log_return, actual_log_return),
        "spearman_ic": safe_spearman_ic(pred_log_return, actual_log_return),
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


# ============================================================
# Optional simple long/short diagnostic
# ============================================================
def long_short_diagnostic(split, metrics, top_fraction=0.10):
    """Rank stocks by predicted next-day log return.

    Research diagnostic only. No costs/slippage/constraints.
    """
    frame = pd.DataFrame({
        "Date": pd.to_datetime(split["target_dates"]),
        "Ticker": split["tickers"],
        "Pred": metrics["pred_log_return"],
        "Actual": split["next_return"],
    })

    daily = []

    for date, day in frame.groupby("Date"):
        day = day.dropna(subset=["Pred", "Actual"])
        if len(day) < 20:
            continue

        n = max(1, int(len(day) * top_fraction))
        day = day.sort_values("Pred")

        short_ret = day.head(n)["Actual"].mean()
        long_ret = day.tail(n)["Actual"].mean()

        daily.append(long_ret - short_ret)

    if len(daily) < 2:
        return None

    daily = np.asarray(daily, dtype=np.float64)
    std = daily.std(ddof=1)

    sharpe = daily.mean() / std * np.sqrt(252.0) if std > 0 else np.nan

    equity = np.cumprod(1.0 + daily)
    running_max = np.maximum.accumulate(equity)
    max_drawdown = np.min(equity / running_max - 1.0)

    return {
        "days": len(daily),
        "mean_daily_spread": float(daily.mean()),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }


# ============================================================
# Result printing
# ============================================================
def print_results(title, split, metrics):
    price = price_metrics(split, metrics)
    naive = naive_baselines(split)
    date_ic = per_date_rank_ic(split, metrics)

    print(f"\n===== {title} =====")
    print(f"Log-return RMSE:    {metrics['rmse_log_return']:.6f}")
    print(f"Log-return MAE:     {metrics['mae_log_return']:.6f}")
    print(f"Price RMSE:         {price['rmse']:.4f}"
          f"   (same-price baseline: {naive['same_price_rmse']:.4f})")
    print(f"Price MAE:          {price['mae']:.4f}"
          f"   (same-price baseline: {naive['same_price_mae']:.4f})")
    print(f"UP/DOWN Accuracy:   {metrics['accuracy']:.2f}%"
          f"   (always-majority baseline: {naive['majority_accuracy']:.2f}%)")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.2f}%   (random: 50.00%)")
    print(f"Direction AUC:      {metrics['auc']:.4f}   (random: 0.5000)")
    print(f"Pooled Pearson IC:  {metrics['pearson_ic']:.4f}")
    print(f"Pooled Spearman IC: {metrics['spearman_ic']:.4f}")
    print(f"Per-date Rank IC:   {date_ic['mean_ic']:.4f}"
          f"   (std {date_ic['ic_std']:.4f}, ICIR {date_ic['icir']:.3f}, "
          f"{date_ic['pct_positive']:.1f}% of {date_ic['days']} days > 0)")

    return price, naive, date_ic


# ============================================================
# Main training
# ============================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Prediction horizon: {HORIZON} trading day")
    print(f"Regression target: {REGRESSION_TARGET}")
    print(f"Direction target: {DIRECTION_TARGET}")

    # --------------------------------------------------------
    # Load + feature engineering + targets
    # --------------------------------------------------------
    df = get_data(horizon=HORIZON)
    df = add_quant_features(df)
    df = add_targets(df, horizon=HORIZON)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------
    features = [
        # Momentum / trend
        "return_1d",
        "return_5d",
        "price_to_MA5",
        "price_to_MA20",
        "MA5_to_MA20",

        # Volatility
        "volatility_5d",
        "volatility_20d",

        # Liquidity
        "volume_change",
        "relative_volume_20d",

        # Seasonality
        "month_sin",
        "month_cos",

        # Technical indicators
        "RSI_14",
        "macd_hist",
        "bollinger_pct_b",

        # Market context
        "excess_return_1d",
        "market_return_1d",
        "market_return_5d",
        "market_volatility",
        "market_trend_20d",
        "market_breadth",

        # Beta / idiosyncratic behavior
        "rolling_beta_20",
        "rolling_beta_60",
        "residual_return_20d",
        "residual_return_60d",
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

    # --------------------------------------------------------
    # Remove invalid rows before fitting scalers.
    # --------------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # --------------------------------------------------------
    # Feature scaler: TRAIN ONLY
    # --------------------------------------------------------
    feature_scaler = StandardScaler()
    train_mask = df["Date"] < VALIDATION_START

    feature_scaler.fit(df.loc[train_mask, features])
    df.loc[:, features] = feature_scaler.transform(df[features])

    # --------------------------------------------------------
    # Regression target scaler: TRAIN ONLY
    # (a target belongs to next_date, so use targets whose target
    # date falls inside the training period)
    # --------------------------------------------------------
    target_scaler = StandardScaler()
    train_target_mask = df["next_date"] < VALIDATION_START
    target_scaler.fit(df.loc[train_target_mask, [REGRESSION_TARGET]])

    df["regression_target_scaled"] = target_scaler.transform(
        df[[REGRESSION_TARGET]]
    ).astype(np.float32)

    # --------------------------------------------------------
    # Training sequences
    # --------------------------------------------------------
    train = create_sequences(
        df,
        features,
        sequence_length=SEQUENCE_LENGTH,
        target_name="regression_target_scaled",
        direction_name=DIRECTION_TARGET,
        max_target_date=VALIDATION_START,
    )

    # --------------------------------------------------------
    # Validation sequences with historical context
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Test sequences with historical context
    # --------------------------------------------------------
    test_context = (
        df[df["Date"] < TEST_START]
        .groupby("Ticker")
        .tail(SEQUENCE_LENGTH)
    )

    test_data = df[df["Date"] >= TEST_START]

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
    )

    print("\n===== DATA SHAPES =====")
    print("X_train:", train["X"].shape)
    print("X_val:", val["X"].shape)
    print("X_test:", test["X"].shape)
    print("Number of features:", len(features))

    # --------------------------------------------------------
    # Direction balance + baselines
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = LSTMModel(
        input_size=len(features),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------
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
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_auc = -np.inf
    best_val_loss = np.inf
    patience_counter = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
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

        # ----------------------------------------------------
        # Validation loss
        # ----------------------------------------------------
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
        date_ic = per_date_rank_ic(val, metrics)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"TrainLoss={train_loss:.5f} "
            f"ValLoss={val_loss:.5f} "
            f"ReturnRMSE={metrics['rmse_log_return']:.6f} "
            f"PriceRMSE={price['rmse']:.4f} "
            f"Acc={metrics['accuracy']:.2f}% "
            f"BalAcc={metrics['balanced_accuracy']:.2f}% "
            f"AUC={metrics['auc']:.4f} "
            f"DateIC={date_ic['mean_ic']:.4f} "
            f"LR={current_lr:.6f}"
        )

        # AUC is more stable than raw accuracy for checkpoint selection.
        improved = False

        if np.isfinite(metrics["auc"]):
            if metrics["auc"] > best_val_auc + 1e-4:
                improved = True
            elif np.isclose(metrics["auc"], best_val_auc, atol=1e-4):
                improved = val_loss < best_val_loss

        if improved:
            best_val_auc = metrics["auc"]
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> Best model saved (validation AUC).")
        else:
            patience_counter += 1
            print(f"  -> No AUC improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------
    print("\nLoading best model...")
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------
    val_metrics = evaluate_model(model, val_loader, target_scaler, device)
    test_metrics = evaluate_model(model, test_loader, target_scaler, device)

    print_results("VALIDATION RESULTS", val, val_metrics)
    print_results("TEST RESULTS", test, test_metrics)

    # --------------------------------------------------------
    # Optional quant diagnostic
    # --------------------------------------------------------
    backtest = long_short_diagnostic(test, test_metrics)

    if backtest is not None:
        print("\n===== OPTIONAL LONG/SHORT DIAGNOSTIC =====")
        print(f"Trading days:      {backtest['days']}")
        print(f"Mean daily spread: {backtest['mean_daily_spread']:.6f}")
        print(f"Annualized Sharpe: {backtest['sharpe']:.3f}")
        print(f"Max drawdown:      {backtest['max_drawdown'] * 100:.2f}%")

    # --------------------------------------------------------
    # Save scaler information needed by inference.
    # --------------------------------------------------------
    scaler_bundle = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": features,
        "sequence_length": SEQUENCE_LENGTH,
        "regression_target": REGRESSION_TARGET,
        "direction_target": DIRECTION_TARGET,
    }

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler_bundle, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scalers saved to: {SCALER_PATH}")

    return model


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train_model()