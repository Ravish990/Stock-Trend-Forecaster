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


def analyze_confidence():
    print("Loading data and model for Confidence Analysis...")

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

    # 3. Get raw predictions on the Test Set
    print("\nRunning inference on the Held-Out Test Set...")
    metrics = run_evaluation(model, test_loader, criterion_ret, criterion_dir, desc="Evaluating")

    # 4. Convert Logits to Probabilities (0.0 to 1.0)
    logits = metrics["pred_logits"]
    actuals = metrics["actual_direction"]
    probabilities = expit(logits)  # Sigmoid function

    print("\n" + "="*60)
    print("CONFIDENCE THRESHOLD ANALYSIS (TEST SET)")
    print("="*60)
    print(f"Total available trades in Test Set: {len(actuals):,}")

    # Test different confidence levels
    thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]

    for t in thresholds:
        # High confidence UP trades (Prob > threshold)
        up_mask = probabilities > t
        
        # High confidence DOWN trades (Prob < 1 - threshold)
        down_mask = probabilities < (1 - t)
        
        # Combine them to see all trades we would actually take
        trade_mask = up_mask | down_mask
        
        trades_taken = np.sum(trade_mask)
        if trades_taken == 0:
            print(f"\nThreshold {t*100:.0f}%: Model never reached this confidence level.")
            continue

        # Calculate accuracy on just these high-confidence trades
        correct_predictions = 0
        
        # Check UP trades
        if np.any(up_mask):
            correct_predictions += np.sum(actuals[up_mask] == 1.0)
            
        # Check DOWN trades
        if np.any(down_mask):
            correct_predictions += np.sum(actuals[down_mask] == 0.0)

        accuracy = (correct_predictions / trades_taken) * 100
        coverage = (trades_taken / len(actuals)) * 100

        print(f"\nThreshold: >{t*100:.0f}% Confidence")
        print(f"  Trades Taken : {trades_taken:,} ({coverage:.1f}% of all opportunities)")
        print(f"  Win Rate     : {accuracy:.2f}%")

if __name__ == "__main__":
    analyze_confidence()