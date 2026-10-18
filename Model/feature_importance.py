"""
Feature importance analysis for the LSTM stock forecaster.
Two methods:
  1. Permutation Importance — shuffle each feature, measure accuracy/IC drop
  2. Gradient Saliency      — average |gradient| w.r.t. each input feature
"""

import random
from collections import namedtuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

from Model.model import LSTMModel
from Model.train_model import (
    DROPOUT, FEATURES, HIDDEN_SIZE, MODEL_PATH,
    NUM_LAYERS, SCALER_PATH, device,
    get_data_with_horizon, prepare_features, prepare_sequences,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ModelData = namedtuple("ModelData", ["model", "X", "y_ret", "y_dir"])


# ------------------------------------------------------------------ load
def load_model_and_data() -> ModelData:
    saved = joblib.load(SCALER_PATH)
    scaler, ret_mean, ret_std = saved["scaler"], saved["ret_mean"], saved["ret_std"]

    data = get_data_with_horizon()
    data, _ = prepare_features(data, scaler=scaler)
    X, y_ret, y_dir, *_ = prepare_sequences(data, split="test")

    y_ret_raw = y_ret * ret_std + ret_mean

    model = LSTMModel(len(FEATURES), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return ModelData(model=model, X=X, y_ret=y_ret_raw, y_dir=y_dir)


def get_predictions(model, X):
    loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=1024)
    preds_ret, preds_dir = [], []
    with torch.inference_mode():
        for (xb,) in loader:
            r, d = model(xb.to(device))
            preds_ret.append(r.cpu().numpy().squeeze())
            preds_dir.append(torch.sigmoid(d).cpu().numpy().squeeze())
    return np.concatenate(preds_ret), np.concatenate(preds_dir)


def ic_score(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 10:
        return 0.0
    return pearsonr(y_true[mask], y_pred[mask])[0]


def direction_accuracy(y_true, y_pred_prob):
    return ((y_pred_prob > 0.5) == y_true).mean()


# ------------------------------------------------------------------ method 1
def permutation_importance(model, X, y_ret, y_dir, n_repeats=3):
    base_ret, base_dir = get_predictions(model, X)
    base_ic  = ic_score(y_ret, base_ret)
    base_acc = direction_accuracy(y_dir, base_dir)

    importances = {}
    for i, feat in enumerate(FEATURES):
        ic_drops, acc_drops = [], []
        for _ in range(n_repeats):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, :, i])
            p_ret, p_dir = get_predictions(model, X_perm)
            ic_drops.append(base_ic  - ic_score(y_ret, p_ret))
            acc_drops.append(base_acc - direction_accuracy(y_dir, p_dir))
        importances[feat] = {
            "ic_drop":  float(np.mean(ic_drops)),
            "acc_drop": float(np.mean(acc_drops)),
        }
    return importances


# ------------------------------------------------------------------ method 2
def gradient_saliency(model, X):
    X_t = torch.tensor(X, requires_grad=True, device=device)
    ret, direction = model(X_t)
    loss = ret.sum() + direction.sum()
    loss.backward()
    saliency = X_t.grad.abs().mean(dim=(0, 1)).cpu().numpy()
    return {feat: float(val) for feat, val in zip(FEATURES, saliency)}


# ------------------------------------------------------------------ plot
def plot_importance(scores: dict, title: str, color: str):
    feats = list(scores.keys())
    vals  = [float(v) for v in scores.values()]
    order = np.argsort(vals)[::-1].tolist()
    feats_sorted = [feats[i] for i in order]
    vals_sorted  = [vals[i]  for i in order]

    plt.figure(figsize=(10, 6))
    plt.barh(feats_sorted[::-1], vals_sorted[::-1], color=color)
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"\n{title}")
    for feat, val in zip(feats_sorted, vals_sorted):
        print(f"  {feat:<25} {val:.5f}")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["permutation", "saliency"], default=None,
                        help="Run only one method. Omit to run both.")
    args = parser.parse_args()

    print("Loading model and test data...")
    md = load_model_and_data()
    print(f"Test samples: {len(md.X)}")

    run_all = args.only is None

    if run_all or args.only == "permutation":
        print("\nComputing Permutation Importance (3 repeats)...")
        perm = permutation_importance(md.model, md.X, md.y_ret, md.y_dir, n_repeats=3)
        plot_importance({f: v["ic_drop"]  for f, v in perm.items()}, "Permutation Importance (IC Drop)",  "steelblue")
        plot_importance({f: v["acc_drop"] for f, v in perm.items()}, "Permutation Importance (Acc Drop)", "darkorange")

    if run_all or args.only == "saliency":
        print("\nComputing Gradient Saliency...")
        saliency = gradient_saliency(md.model, md.X)
        plot_importance(saliency, "Gradient Saliency", "seagreen")
