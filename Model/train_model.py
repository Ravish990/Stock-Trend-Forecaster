import copy
import os
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
DROPOUT = 0.4

WEIGHT_DECAY = 3e-4
LEARNING_RATE = 3e-4
ALPHA = 1.0          
BETA = 1.0           
EPOCHS = 50
BATCH_SIZE = 2048
PATIENCE = 8         
SCHED_PATIENCE = 3

MIN_TICKERS_PER_DATE = 50
OOS_PREDS_OUTPUT_PATH = "lstm_oos_predictions.csv"

SCALER_DIR = "scalers"
MODEL_DIR = "saved_models"

FEATURES = [
    "return_1d", "return_5d", "return_20d", 
    "mom_12_1", "mom_6_1",
    "price_to_MA5", "price_to_MA20",
    "volatility_5d", "volatility_60d",
    "relative_volume_20", "RSI_14", "bollinger_pct_b", 
    "excess_return_1d",
    "rolling_beta_20", "residual_return_1d", "atr_14", "dist_52w_high",
    "dist_52w_low",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def build_model():
    return LSTMModel(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)


def get_data_with_horizon():
    from Data_Processing.target_data import get_data
    data = get_data(horizon=HORIZON)
    data = data[data["Date"] >= pd.Timestamp("2002-01-01")].copy()

    if "next_date" not in data.columns:
        data["next_date"] = data.groupby("Ticker")["Date"].shift(-HORIZON)

    data["Ticker"] = data["Ticker"].astype(str)
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    data = add_cross_sectional_rank_target(data)

    return data


def add_cross_sectional_rank_target(data):
    data = data.copy()
    raw = pd.to_numeric(data["next_log_return"], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)

    grp = raw.groupby(data["Date"])
    count = grp.transform("count")
    
    ok = raw.notna() & (count >= MIN_TICKERS_PER_DATE)
    
    data["next_log_return_raw"] = raw
    data["next_log_return"] = np.where(ok, grp.rank(pct=True), np.nan).astype(np.float32)
    data["direction_target"] = np.where(ok, (data["next_log_return"] >= 0.5).astype(np.float32), np.nan)
    return data


def generate_walk_forward_folds(data, start_test_year, end_test_year):
    for test_year in range(start_test_year, end_test_year + 1):
        val_year = test_year - 1
        
        train_mask = data["Date"].dt.year < val_year
        val_mask = data["Date"].dt.year == val_year
        test_mask = data["Date"].dt.year == test_year
        
        fold_data = data.copy()
        
        g = fold_data.groupby("Date")[FEATURES]
        fold_data[FEATURES] = (fold_data[FEATURES] - g.transform("mean")) / (g.transform("std") + 1e-8)
        fold_data[FEATURES] = fold_data[FEATURES].fillna(0.0)
        
        scaler_fit_data = fold_data.loc[train_mask, FEATURES]
        scaler = StandardScaler()
        scaler.fit(scaler_fit_data)
        fold_data.loc[:, FEATURES] = scaler.transform(fold_data[FEATURES]).astype(np.float32)
        
        scaler_path = f"{SCALER_DIR}/scaler_{test_year}.pkl"
        joblib.dump(scaler, scaler_path)
        
        train_seq = prepare_sequences(fold_data[fold_data["Date"].dt.year <= val_year], train_mask)
        val_seq = prepare_sequences(fold_data[fold_data["Date"].dt.year <= test_year], val_mask)
        test_seq = prepare_sequences(fold_data, test_mask)
        
        yield train_seq, val_seq, test_seq, scaler_path, test_year


def prepare_sequences(data, date_mask):
    X, y_return, y_direction = [], [], []
    input_dates, target_dates, tickers = [], [], []

    for ticker, group in data.groupby("Ticker", sort=False):
        group = group.sort_values("Date").reset_index(drop=False)
        feature_array = group[FEATURES].to_numpy(dtype=np.float32)
        returns = group["next_log_return"].to_numpy(dtype=np.float32)
        directions = group["direction_target"].to_numpy(dtype=np.float32)
        dates = group["Date"].to_numpy()
        next_dates = group["next_date"].to_numpy()
        orig_indices = group["index"].to_numpy()

        for i in range(SEQUENCE - 1, len(group)):
            if not date_mask.loc[orig_indices[i]]:
                continue
                
            if not np.isfinite(returns[i]) or pd.isna(next_dates[i]):
                continue

            window = feature_array[i - SEQUENCE + 1 : i + 1]
            if window.shape != (SEQUENCE, len(FEATURES)):
                continue

            X.append(window)
            y_return.append(returns[i])
            y_direction.append(directions[i])
            input_dates.append(pd.Timestamp(dates[i]))
            target_dates.append(pd.Timestamp(next_dates[i]))
            tickers.append(ticker)

    if not X:
        return None

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_return, dtype=np.float32),
        np.asarray(y_direction, dtype=np.float32),
        np.asarray(input_dates),
        np.asarray(target_dates),
        np.asarray(tickers),
    )


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
    all_predictions, all_returns = [], []

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

            all_predictions.append(pred_ret.cpu().numpy())
            all_returns.append(y_ret.cpu().numpy())

    return {
        "loss": total_loss / total_samples,
        "pred_return": np.concatenate(all_predictions),
        "actual_return": np.concatenate(all_returns),
    }


def daily_ic(pred, actual, dates, min_names=10):
    df = pd.DataFrame({"d": dates, "p": pred, "a": actual})
    df["pr"] = df.groupby("d")["p"].rank()
    df["ar"] = df.groupby("d")["a"].rank()

    ics = []
    for _, g in df.groupby("d"):
        if len(g) < min_names or g["pr"].nunique() < 2:
            continue
        ics.append(g["pr"].corr(g["ar"]))
    ics = np.asarray(ics, dtype=np.float64)
    return ics[np.isfinite(ics)]


def train_walk_forward():
    print("\nLoading dataset...")
    data = get_data_with_horizon()
    
    start_test_year = 2018
    end_test_year = 2025
    
    all_oos_predictions = []
    
    criterion_ret = nn.MSELoss() 
    criterion_dir = nn.BCEWithLogitsLoss()
    
    for train_data, val_data, test_data, scaler_path, test_year in generate_walk_forward_folds(data, start_test_year, end_test_year):
        print(f"\n{'='*70}\nWALK-FORWARD FOLD: TEST YEAR {test_year}\n{'='*70}")
        
        if not train_data or not val_data or not test_data:
            print(f"Skipping fold {test_year} due to insufficient data.")
            continue
            
        train_loader = create_dataloader(train_data, shuffle=True)
        val_loader = create_dataloader(val_data, shuffle=False)
        test_loader = create_dataloader(test_data, shuffle=False)
        
        model = build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=SCHED_PATIENCE, min_lr=1e-6)
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for X_batch, y_ret, y_dir in tqdm(train_loader, desc=f"Train E{epoch}", leave=False):
                optimizer.zero_grad(set_to_none=True)
                pred_ret, pred_dir = model(X_batch.to(device))
                loss = ALPHA * criterion_ret(pred_ret.reshape(-1), y_ret.to(device)) + BETA * criterion_dir(pred_dir.reshape(-1), y_dir.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            val_metrics = run_evaluation(model, val_loader, criterion_ret, criterion_dir, desc=f"Val E{epoch}")
            val_loss = val_metrics["loss"]
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break
                
        model.load_state_dict(best_state)
        
        model_save_path = f"{MODEL_DIR}/lstm_fold_{test_year}.pth"
        torch.save(best_state, model_save_path)
        print(f"Saved model for fold {test_year} to {model_save_path}")

        test_metrics = run_evaluation(model, test_loader, criterion_ret, criterion_dir, desc="Test")
        
        test_dates, test_tickers = test_data[3], test_data[5]
        fold_df = pd.DataFrame({
            "Date": test_dates,
            "Ticker": test_tickers,
            "lstm_pred": test_metrics["pred_return"],
            "actual_rank": test_metrics["actual_return"]
        })
        all_oos_predictions.append(fold_df)
        
        ret_ic = daily_ic(test_metrics["pred_return"], test_metrics["actual_return"], test_dates)
        print(f"Test Year {test_year} IC: {ret_ic.mean():.4f}")

    final_oos_df = pd.concat(all_oos_predictions, ignore_index=True)
    final_oos_df.to_csv(OOS_PREDS_OUTPUT_PATH, index=False)
    print(f"\nSaved OOS predictions to {OOS_PREDS_OUTPUT_PATH}")



if __name__ == "__main__":
    train_walk_forward()