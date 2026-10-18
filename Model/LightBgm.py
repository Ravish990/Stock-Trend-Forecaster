"""
LightGBM baseline with target ranking, seed averaging, and Walk-Forward CV.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.inspection import permutation_importance

from Model.train_model import (
    FEATURES, SEED, prepare_data_for_training, daily_ic, ic_summary,
)

TRAIN_DATE_STRIDE = 2
TARGET_CLIP = 4.0
SEEDS = [SEED, SEED + 42, SEED + 123]

COMMON_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=2000,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=10.0,
    n_jobs=-1,
    verbose=-1,
)

def subsample_by_date(X, y_ret, y_dir, dates, stride):
    if stride <= 1: return X, y_ret, y_dir, dates
    keep_dates = np.unique(dates)[::stride]
    mask = np.isin(dates, keep_dates)
    return X[mask], y_ret[mask], y_dir[mask], dates[mask]

def make_tabular(X):
    parts = [
        X[:, -1, :], X[:, -2, :], X[:, -5, :], X[:, 0, :],
        X.mean(axis=1), X.std(axis=1),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)

def tabular_names():
    suffixes = ["t", "t1", "t4", "t9", "mean10", "std10"]
    return [f"{f}_{s}" for s in suffixes for f in FEATURES]

def convert_to_cross_sectional_rank(target_vals, dates):
    df = pd.DataFrame({"d": dates, "y": target_vals})
    return df.groupby("d")["y"].rank(pct=True).to_numpy()

def quintile_spread(score, actual, dates):
    df = pd.DataFrame({"d": dates, "s": score, "a": actual})
    pct = df.groupby("d")["s"].rank(method="first", pct=True)
    df["q"] = (np.ceil(pct * 5) - 1).astype(int).clip(0, 4)
    top = df[df.q == 4].groupby("d")["a"].mean()
    bot = df[df.q == 0].groupby("d")["a"].mean()
    spread = (top - bot).dropna()
    return float(spread.mean()), float((spread > 0).mean())

def combined_score(pred_ret, prob_up, dates):
    df = pd.DataFrame({"d": dates, "r": pred_ret, "p": prob_up})
    r = df.groupby("d")["r"].rank(pct=True)
    p = df.groupby("d")["p"].rank(pct=True)
    return ((r + p) / 2.0).to_numpy()

def report(name, score, actual_scaled, dates, artifacts):
    ic = ic_summary(daily_ic(score, actual_scaled, dates))
    actual_real = actual_scaled * artifacts["ret_std"] + artifacts["ret_mean"]
    spread, spread_pos = quintile_spread(score, actual_real, dates)
    print(f"{name:<22} IC {ic['mean']:+.4f} | adj t {ic['t']:+.2f} | "
          f">0 days {ic['pos']:.1%} | spread {spread:+.5f} ({spread_pos:.1%} days >0)")
    return ic["mean"]

def get_ensemble_predictions(models, X, best_iters, is_proba=False):
    preds = []
    for model, it in zip(models, best_iters):
        if is_proba:
            preds.append(model.predict_proba(X, num_iteration=it)[:, 1])
        else:
            preds.append(model.predict(X, num_iteration=it))
    return np.mean(preds, axis=0)

def generate_walk_forward_folds(X, y_ret, y_dir, dates, min_train_years=4):
    """
    Yields walk-forward splits: 
    - Train: Year < N-1
    - Eval: Year == N-1
    - Test: Year == N
    """
    dates_pd = pd.to_datetime(dates)
    years = dates_pd.year
    unique_years = np.sort(np.unique(years))

    for i in range(min_train_years + 1, len(unique_years)):
        test_year = unique_years[i]
        eval_year = unique_years[i-1]

        train_mask = years < eval_year
        eval_mask = years == eval_year
        test_mask = years == test_year

        yield test_year, (
            (X[train_mask], y_ret[train_mask], y_dir[train_mask], dates[train_mask]),
            (X[eval_mask], y_ret[eval_mask], y_dir[eval_mask], dates[eval_mask]),
            (X[test_mask], y_ret[test_mask], y_dir[test_mask], dates[test_mask])
        )

def main():
    train_data, eval_data, test_data, artifacts = prepare_data_for_training()
    
    # 1. Recombine all pre-processed data to create a continuous timeline
    X_all = np.concatenate([train_data[0], eval_data[0], test_data[0]], axis=0)
    y_ret_all = np.concatenate([train_data[1], eval_data[1], test_data[1]], axis=0)
    y_dir_all = np.concatenate([train_data[2], eval_data[2], test_data[2]], axis=0)
    dates_all = np.concatenate([train_data[3], eval_data[3], test_data[3]], axis=0)

    # Ensure chronological order
    sort_idx = np.argsort(dates_all)
    X_all, y_ret_all, y_dir_all, dates_all = (
        X_all[sort_idx], y_ret_all[sort_idx], y_dir_all[sort_idx], dates_all[sort_idx]
    )

    X_tab_all = make_tabular(X_all)
    
    oof_te_r, oof_te_p, oof_te_c, oof_te_y, oof_te_dates = [], [], [], [], []

    print("\nStarting Walk-Forward Cross-Validation...")
    folds = generate_walk_forward_folds(X_tab_all, y_ret_all, y_dir_all, dates_all, min_train_years=4)

    for test_year, (tr, ev, te) in folds:
        print(f"\n--- Fold: Test Year {test_year} ---")
        
        # Subsample training dates for speed and decorrelation
        X_tr, y_tr_ret, y_tr_dir, d_tr = subsample_by_date(*tr, TRAIN_DATE_STRIDE)
        X_ev, y_ev_ret, y_ev_dir, d_ev = ev
        X_te, y_te_ret, y_te_dir, d_te = te

        # Convert targets to daily percentile ranks
        y_tr_rank = convert_to_cross_sectional_rank(np.clip(y_tr_ret, -TARGET_CLIP, TARGET_CLIP), d_tr)
        y_ev_rank = convert_to_cross_sectional_rank(y_ev_ret, d_ev)
        
        # Train Regressor (Seed Averaged)
        reg_models, reg_best_iters = [], []
        for s in SEEDS:
            reg = lgb.LGBMRegressor(objective="regression", random_state=s, **COMMON_PARAMS)
            reg.fit(X_tr, y_tr_rank, eval_set=[(X_ev, y_ev_rank)], callbacks=[lgb.early_stopping(50, verbose=False)])
            reg_best_iters.append(max(int(reg.best_iteration_ or 1), 1))
            reg_models.append(reg)

        # Train Classifier (Seed Averaged)
        clf_models, clf_best_iters = [], []
        for s in SEEDS:
            clf = lgb.LGBMClassifier(objective="binary", random_state=s, **COMMON_PARAMS)
            clf.fit(X_tr, y_tr_dir, eval_set=[(X_ev, y_ev_dir)], callbacks=[lgb.early_stopping(50, verbose=False)])
            clf_best_iters.append(max(int(clf.best_iteration_ or 1), 1))
            clf_models.append(clf)

        # Generate Test Predictions
        te_r = get_ensemble_predictions(reg_models, X_te, reg_best_iters)
        te_p = get_ensemble_predictions(clf_models, X_te, clf_best_iters, is_proba=True)
        te_c = combined_score(te_r, te_p, d_te)

        # Print per-fold quick IC
        fold_ic = report(f"Year {test_year} Combined", te_c, y_te_ret, d_te, artifacts)

        # Store Out-of-Sample predictions
        oof_te_r.append(te_r)
        oof_te_p.append(te_p)
        oof_te_c.append(te_c)
        oof_te_y.append(y_te_ret)
        oof_te_dates.append(d_te)

    # 3. Concatenate and Evaluate Total Out-Of-Sample Results
    final_r = np.concatenate(oof_te_r)
    final_p = np.concatenate(oof_te_p)
    final_c = np.concatenate(oof_te_c)
    final_y = np.concatenate(oof_te_y)
    final_dates = np.concatenate(oof_te_dates)

    print("\n" + "=" * 100)
    print("TOTAL WALK-FORWARD OUT-OF-SAMPLE EVALUATION")
    print("=" * 100)
    report("LGBM return head (rank)", final_r, final_y, final_dates, artifacts)
    report("LGBM direction head", final_p, final_y, final_dates, artifacts)
    report("LGBM combined (rank avg)", final_c, final_y, final_dates, artifacts)

if __name__ == "__main__":
    main()