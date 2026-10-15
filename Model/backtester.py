import numpy as np
import torch
from scipy.special import expit
from torch import nn

from Model.model import LSTMModel
from Model.train_model import (
    DROPOUT,
    FEATURES,
    HIDDEN_SIZE,
    MODEL_PATH,
    NUM_LAYERS,
    create_dataloader,
    device,
    get_data_with_horizon,
    prepare_features,
    prepare_sequences,
    run_evaluation,
)


def run_backtest():
    print("Loading data and model for Backtesting...")

    # 1. Load Data
    data = get_data_with_horizon()
    data, _ = prepare_features(data)
    test_data = prepare_sequences(data, "test")
    test_loader = create_dataloader(test_data, shuffle=False)

    # 2. Load Saved Model
    model = LSTMModel(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    criterion_ret = nn.MSELoss()
    criterion_dir = nn.BCEWithLogitsLoss()

    # 3. Get predictions on the Test Set
    print("\nRunning inference on the Held-Out Test Set...")
    metrics = run_evaluation(model, test_loader, criterion_ret, criterion_dir, desc="Evaluating")

    probabilities = expit(metrics["pred_logits"])
    actual_log_returns = metrics["actual_return"]  # The actual 5-day log returns
    
    print("\n" + "="*70)
    print("STRATEGY BACKTEST: LONG/SHORT PORTFOLIO (TEST SET)")
    print("="*70)
    
    # Baseline: If we just bought every single stock for 5 days
    baseline_avg_return = np.mean(actual_log_returns)
    print(f"BASELINE (Buy & Hold Everything):")
    print(f"  Average 5-Day Return per Trade : {baseline_avg_return * 100:.3f}%\n")
    print("-" * 70)
    
    thresholds = [0.50, 0.54, 0.56, 0.58, 0.60]
    
    for t in thresholds:
        # Long trades (Model is > threshold confident it goes UP)
        long_mask = probabilities > t
        
        # Short trades (Model is > threshold confident it goes DOWN)
        short_mask = probabilities < (1 - t)
        
        total_trades = np.sum(long_mask) + np.sum(short_mask)
        if total_trades == 0:
            continue
            
        # For Longs, we get the actual return. For Shorts, we get the inverted return.
        long_returns = actual_log_returns[long_mask]
        short_returns = -actual_log_returns[short_mask]
        
        # Combine all trade returns for this threshold
        strategy_returns = np.concatenate([long_returns, short_returns])
        
        avg_trade_ret = np.mean(strategy_returns)
        win_rate = np.mean(strategy_returns > 0) * 100
        
        # Subtract a 5 basis point (0.05%) friction fee per trade for slippage/commissions
        avg_trade_ret_after_fees = avg_trade_ret - 0.0005
        
        print(f"Threshold: >{t*100:.0f}% Confidence")
        print(f"  Total Trades Taken       : {total_trades:,} (Longs: {np.sum(long_mask):,}, Shorts: {np.sum(short_mask):,})")
        print(f"  Win Rate                 : {win_rate:.2f}%")
        print(f"  Gross Avg Return / Trade : {avg_trade_ret * 100:.3f}%")
        print(f"  Net Return (After Fees)  : {avg_trade_ret_after_fees * 100:.3f}%")
        print("-" * 70)

if __name__ == "__main__":
    run_backtest()