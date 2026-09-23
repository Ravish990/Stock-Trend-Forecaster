# ============================================================
# Load Trained LSTM Model and Evaluate on Test Data
# ============================================================

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from Data_Processing.target_data import get_data
from Model.model import LSTMModel
import matplotlib.pyplot as plt


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Create Sequences
# Same logic used during training (train_model.py)
# ============================================================

def create_sequences_with_date_filter(
    df,
    features,
    sequence_length=20,
    min_target_date=None,
    max_target_date=None
):

    X_sequences = []
    return_targets = []
    direction_targets = []

    # Also keep the raw current-day Close price for each
    # sequence, so we can reconstruct a price-level plot later
    # (purely for visualization — the model itself never sees
    # or predicts raw price).
    current_close_prices = []

    for ticker, group in df.groupby("Ticker"):

        group = group.sort_values("Date").reset_index(drop=True)

        X = group[features].values

        returns = group["next_return"].values
        directions = group["target"].values
        target_dates = group["next_date"].values
        closes = group["Close"].values

        for i in range(sequence_length, len(group)):

            target_date = target_dates[i]

            if pd.isna(target_date):
                continue

            if min_target_date is not None:
                if target_date < np.datetime64(min_target_date):
                    continue

            if max_target_date is not None:
                if target_date >= np.datetime64(max_target_date):
                    continue

            X_sequences.append(
                X[i - sequence_length:i]
            )

            return_targets.append(
                returns[i]
            )

            direction_targets.append(
                directions[i]
            )

            current_close_prices.append(
                closes[i]
            )

    return (
        np.array(X_sequences),
        np.array(return_targets),
        np.array(direction_targets),
        np.array(current_close_prices)
    )


# ============================================================
# Naive Baseline (predict 0 return / majority class)
# ============================================================

def naive_baseline(return_targets, direction_targets):

    naive_rmse = np.sqrt(
        np.mean(return_targets ** 2)
    )

    positive_rate = direction_targets.mean()
    majority_accuracy = max(
        positive_rate,
        1 - positive_rate
    ) * 100

    return naive_rmse, majority_accuracy


# ============================================================
# Evaluation Function
# Same metrics used during training
# ============================================================

def evaluate_model(
    model,
    data_loader,
    plot=False,
    n_points=200
):
    model.eval()

    total_squared_error = 0.0

    total_correct = 0
    total_samples = 0

    # Store predictions for the graph (in return space)
    actual_returns = []
    predicted_returns = []

    actual_directions = []
    predicted_directions = []

    current_closes_all = []

    with torch.no_grad():

        for (
            X_batch,
            return_batch,
            direction_batch,
            close_batch
        ) in data_loader:

            # ---------------------------------------------
            # Move data to CPU/GPU
            # ---------------------------------------------

            X_batch = X_batch.to(device)
            return_batch = return_batch.to(device)
            direction_batch = direction_batch.to(device)

            # ---------------------------------------------
            # Model prediction
            # ---------------------------------------------

            return_pred, direction_pred = model(X_batch)

            return_pred = return_pred.squeeze(1)
            direction_pred = direction_pred.squeeze(1)

            # ---------------------------------------------
            # Return error
            # ---------------------------------------------

            squared_error = (
                return_pred - return_batch
            ) ** 2

            total_squared_error += (
                squared_error.sum().item()
            )

            # ---------------------------------------------
            # Store return predictions
            # ---------------------------------------------

            actual_returns.extend(
                return_batch.cpu().numpy()
            )

            predicted_returns.extend(
                return_pred.cpu().numpy()
            )

            current_closes_all.extend(
                close_batch.numpy()
            )

            # ---------------------------------------------
            # Direction prediction
            # ---------------------------------------------

            direction_probability = torch.sigmoid(
                direction_pred
            )

            direction_prediction = (
                direction_probability >= 0.5
            ).float()

            total_correct += (
                direction_prediction == direction_batch
            ).sum().item()

            total_samples += (
                direction_batch.size(0)
            )

            actual_directions.extend(
                direction_batch.cpu().numpy()
            )

            predicted_directions.extend(
                direction_prediction.cpu().numpy()
            )

    # -----------------------------------------------------
    # Convert lists to NumPy arrays
    # -----------------------------------------------------

    actual_returns = np.array(actual_returns)
    predicted_returns = np.array(predicted_returns)
    actual_directions = np.array(actual_directions)
    predicted_directions = np.array(predicted_directions)
    current_closes_all = np.array(current_closes_all)

    # -----------------------------------------------------
    # Calculate RMSE (in return space)
    # -----------------------------------------------------

    rmse = np.sqrt(
        total_squared_error / total_samples
    )

    # -----------------------------------------------------
    # Calculate directional accuracy
    # -----------------------------------------------------

    directional_accuracy = (
        total_correct / total_samples
    ) * 100

    # -----------------------------------------------------
    # Naive baseline, for context
    # -----------------------------------------------------

    naive_rmse, naive_acc = naive_baseline(
        actual_returns, actual_directions
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print(f"Test Return RMSE: {rmse:.5f}")
    print(f"Test Directional Accuracy: {directional_accuracy:.2f}%")
    print(
        f"Naive baseline -> RMSE: {naive_rmse:.5f}  "
        f"Majority-class Acc: {naive_acc:.2f}%"
    )

    # -----------------------------------------------------
    # Plot Actual vs Predicted
    #
    # Reconstructed into price space purely for a readable
    # chart: price_next = current_close * (1 + predicted_return)
    # This is NOT what the model was trained/evaluated on —
    # the real metrics above are computed in return space.
    # -----------------------------------------------------

    if plot:

        n_points = min(n_points, len(actual_returns))

        actual_price_reconstructed = (
            current_closes_all[:n_points] *
            (1 + actual_returns[:n_points])
        )

        predicted_price_reconstructed = (
            current_closes_all[:n_points] *
            (1 + predicted_returns[:n_points])
        )

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Return-space plot (the metric that actually matters)
        axes[0].plot(
            actual_returns[:n_points],
            label="Actual Return"
        )
        axes[0].plot(
            predicted_returns[:n_points],
            label="Predicted Return"
        )
        axes[0].axhline(0, color="gray", linewidth=0.8)
        axes[0].set_xlabel("Test Sample")
        axes[0].set_ylabel("Next-Day Return")
        axes[0].set_title("Actual vs Predicted Return (model's real target)")
        axes[0].legend()
        axes[0].grid(True)

        # Reconstructed price-space plot (for intuition only)
        axes[1].plot(
            actual_price_reconstructed,
            label="Actual Price"
        )
        axes[1].plot(
            predicted_price_reconstructed,
            label="Predicted Price"
        )
        axes[1].set_xlabel("Test Sample")
        axes[1].set_ylabel("Stock Price")
        axes[1].set_title(
            "Reconstructed Price (for reference only — not the training target)"
        )
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------
    # Return everything
    # -----------------------------------------------------

    return (
        rmse,
        directional_accuracy,
        naive_rmse,
        naive_acc,
        actual_returns,
        predicted_returns,
        actual_directions,
        predicted_directions
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 1. Load data
    # ========================================================

    print("\nLoading data...")

    df = get_data()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    # ========================================================
    # 2. Create next_date
    # ========================================================

    df["next_date"] = (
        df.groupby("Ticker")["Date"]
        .shift(-1)
    )

    # ========================================================
    # 3. Features
    # EXACTLY same as train_model.py (relative MA features,
    # not raw MA5 / MA20)
    # ========================================================

    features = [
        "return_1d",
        "return_5d",
        "price_to_MA5",
        "price_to_MA20",
        "MA5_to_MA20",
        "volatility_5d",
        "volume_change",
        "month_sin",
        "month_cos"
    ]

    # ========================================================
    # 4. Split dates
    # EXACTLY same as train_model.py
    # ========================================================

    validation_start = pd.Timestamp("2012-01-01")
    test_start = pd.Timestamp("2015-01-01")

    # ========================================================
    # 5. Recreate the scaler
    #
    # IMPORTANT:
    # Your original training code fitted the scaler ONLY
    # using data before 2012-01-01. We reproduce that here
    # because scaler.pkl was not saved.
    # ========================================================

    print("\nRecreating StandardScaler...")

    train_rows = df[
        df["Date"] < validation_start
    ].copy()

    scaler = StandardScaler()

    scaler.fit(train_rows[features])

    print("Scaler recreated successfully.")

    # ========================================================
    # 6. Apply scaler to entire dataset
    # ========================================================

    df[features] = scaler.transform(df[features])

    # ========================================================
    # 7. Create test context
    #
    # Same logic as train_model.py: last 20 observations
    # before 2015 for every ticker.
    # ========================================================

    print("\nCreating test context...")

    test_context = (
        df[df["Date"] < test_start]
        .groupby("Ticker")
        .tail(20)
    )

    # ========================================================
    # 8. Get test data
    # ========================================================

    test_data = df[df["Date"] >= test_start]

    # ========================================================
    # 9. Combine context + test data
    # ========================================================

    test_sequence_df = pd.concat(
        [test_context, test_data]
    )

    # ========================================================
    # 10. Sort test data
    # ========================================================

    test_sequence_df = (
        test_sequence_df
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    # ========================================================
    # 11. Create test sequences
    # ========================================================

    print("\nCreating test sequences...")

    X_test, return_test, direction_test, close_test = (
        create_sequences_with_date_filter(
            test_sequence_df,
            features,
            sequence_length=20,
            min_target_date=test_start,
            max_target_date=None
        )
    )

    # ========================================================
    # 12. Print shapes
    # ========================================================

    print("\n===== TEST DATA SHAPES =====")
    print("X_test:", X_test.shape)
    print("Return test:", return_test.shape)
    print("Direction test:", direction_test.shape)

    # ========================================================
    # 13. Convert to PyTorch tensors
    # ========================================================

    X_test = torch.tensor(X_test, dtype=torch.float32)
    return_test = torch.tensor(return_test, dtype=torch.float32)
    direction_test = torch.tensor(direction_test, dtype=torch.float32)
    close_test = torch.tensor(close_test, dtype=torch.float32)

    # ========================================================
    # 14. Create test dataset
    # ========================================================

    test_dataset = TensorDataset(
        X_test,
        return_test,
        direction_test,
        close_test
    )

    # ========================================================
    # 15. Create DataLoader
    # Same batch size as training
    # ========================================================

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # ========================================================
    # 16. Create model architecture
    # MUST exactly match training
    # ========================================================

    print("\nCreating LSTM model...")

    model = LSTMModel(
        input_size=len(features),
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )

    # ========================================================
    # 17. Load saved model
    # ========================================================

    model_path = "best_lstm_stock_model.pth"

    print("\nLoading trained model...")

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    # ========================================================
    # 18. Evaluate
    # ========================================================

    print("\nEvaluating test data...")

    (
        rmse,
        accuracy,
        naive_rmse,
        naive_acc,
        actual_returns,
        predicted_returns,
        actual_directions,
        predicted_directions
    ) = evaluate_model(
        model,
        test_loader,
        plot=True,
        n_points=200
    )

    # ========================================================
    # 19. Final results
    # ========================================================

    print("\n==========================================")
    print("           FINAL TEST RESULTS")
    print("==========================================")
    print(f"Test Return RMSE: {rmse:.5f}")
    print(f"Test Directional Accuracy: {accuracy:.2f}%")
    print(
        f"Naive Baseline RMSE: {naive_rmse:.5f}  "
        f"Naive Majority-Class Accuracy: {naive_acc:.2f}%"
    )
    print("==========================================")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()