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


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Create Sequences
# Same logic used during training
# ============================================================

def create_sequences_with_date_filter(
    df,
    features,
    sequence_length=20,
    min_target_date=None,
    max_target_date=None
):

    X_sequences = []
    price_targets = []
    direction_targets = []

    # Process every stock separately
    for ticker, group in df.groupby("Ticker"):

        # Sort by date
        group = group.sort_values(
            "Date"
        ).reset_index(drop=True)

        X = group[features].values

        prices = group["next_close"].values

        directions = group["target"].values

        target_dates = group["next_date"].values

        # Start after enough observations exist
        for i in range(
            sequence_length,
            len(group)
        ):

            target_date = target_dates[i]

            # ------------------------------------------------
            # Filter target date
            # ------------------------------------------------

            if min_target_date is not None:

                if target_date < np.datetime64(
                    min_target_date
                ):
                    continue

            if max_target_date is not None:

                if target_date >= np.datetime64(
                    max_target_date
                ):
                    continue

            # ------------------------------------------------
            # Create 20-day sequence
            # ------------------------------------------------

            X_sequences.append(
                X[
                    i - sequence_length:i
                ]
            )

            price_targets.append(
                prices[i]
            )

            direction_targets.append(
                directions[i]
            )

    return (
        np.array(X_sequences),
        np.array(price_targets),
        np.array(direction_targets)
    )


# ============================================================
# Evaluation Function
# Same metrics used during training
# ============================================================

def evaluate_model(
    model,
    data_loader
):

    model.eval()

    total_squared_error = 0.0

    total_correct = 0

    total_samples = 0

    with torch.no_grad():

        for (
            X_batch,
            price_batch,
            direction_batch
        ) in data_loader:

            # Move data to CPU/GPU
            X_batch = X_batch.to(device)

            price_batch = price_batch.to(device)

            direction_batch = direction_batch.to(device)

            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            price_pred, direction_pred = model(
                X_batch
            )

            price_pred = price_pred.squeeze(1)

            direction_pred = direction_pred.squeeze(1)

            # ------------------------------------------------
            # Price error
            # ------------------------------------------------

            squared_error = (
                price_pred - price_batch
            ) ** 2

            total_squared_error += (
                squared_error.sum().item()
            )

            # ------------------------------------------------
            # Direction prediction
            # ------------------------------------------------

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

    # --------------------------------------------------------
    # Calculate RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        total_squared_error /
        total_samples
    )

    # --------------------------------------------------------
    # Calculate directional accuracy
    # --------------------------------------------------------

    directional_accuracy = (
        total_correct /
        total_samples
    ) * 100

    return rmse, directional_accuracy


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 1. Load data
    # ========================================================

    print("\nLoading data...")

    df = get_data()

    # Convert Date column
    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # Sort exactly like training
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
    # EXACTLY same as train_model.py
    # ========================================================

    features = [
        "return_1d",
        "return_5d",
        "MA5",
        "MA20",
        "volatility_5d",
        "volume_change",
        "month_sin",
        "month_cos"
    ]


    # ========================================================
    # 4. Split dates
    # EXACTLY same as train_model.py
    # ========================================================

    validation_start = pd.Timestamp(
        "2012-01-01"
    )

    test_start = pd.Timestamp(
        "2015-01-01"
    )


    # ========================================================
    # 5. Recreate the scaler
    #
    # IMPORTANT:
    # Your original training code fitted the scaler ONLY
    # using data before 2012-01-01.
    #
    # We reproduce that here because you did not save
    # scaler.pkl.
    # ========================================================

    print(
        "\nRecreating StandardScaler..."
    )

    train_rows = df[
        df["Date"] < validation_start
    ].copy()

    scaler = StandardScaler()

    scaler.fit(
        train_rows[features]
    )

    print(
        "Scaler recreated successfully."
    )


    # ========================================================
    # 6. Apply scaler to entire dataset
    # ========================================================

    df[features] = scaler.transform(
        df[features]
    )


    # ========================================================
    # 7. Create test context
    #
    # Same logic as train_model.py:
    # Take the last 20 observations before 2015
    # for every ticker.
    # ========================================================

    print(
        "\nCreating test context..."
    )

    test_context = (
        df[
            df["Date"] < test_start
        ]
        .groupby("Ticker")
        .tail(20)
    )


    # ========================================================
    # 8. Get test data
    # ========================================================

    test_data = df[
        df["Date"] >= test_start
    ]


    # ========================================================
    # 9. Combine context + test data
    # ========================================================

    test_sequence_df = pd.concat(
        [
            test_context,
            test_data
        ]
    )


    # ========================================================
    # 10. Sort test data
    # ========================================================

    test_sequence_df = (
        test_sequence_df
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # 11. Create test sequences
    # ========================================================

    print(
        "\nCreating test sequences..."
    )

    X_test, price_test, direction_test = (
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

    print(
        "\n===== TEST DATA SHAPES ====="
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "Price test:",
        price_test.shape
    )

    print(
        "Direction test:",
        direction_test.shape
    )


    # ========================================================
    # 13. Convert to PyTorch tensors
    # ========================================================

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    price_test = torch.tensor(
        price_test,
        dtype=torch.float32
    )

    direction_test = torch.tensor(
        direction_test,
        dtype=torch.float32
    )


    # ========================================================
    # 14. Create test dataset
    # ========================================================

    test_dataset = TensorDataset(
        X_test,
        price_test,
        direction_test
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

    print(
        "\nCreating LSTM model..."
    )

    model = LSTMModel(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )


    # ========================================================
    # 17. Load saved model
    # ========================================================

    model_path = (
        "best_lstm_stock_model.pth"
    )

    print(
        "\nLoading trained model..."
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)

    model.eval()

    print(
        "Model loaded successfully!"
    )


    # ========================================================
    # 18. Evaluate
    # ========================================================

    print(
        "\nEvaluating test data..."
    )

    test_rmse, test_accuracy = (
        evaluate_model(
            model,
            test_loader
        )
    )


    # ========================================================
    # 19. Final results
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "           FINAL TEST RESULTS"
    )

    print(
        "=========================================="
    )

    print(
        f"Test Price RMSE: "
        f"{test_rmse:.4f}"
    )

    print(
        f"Test Directional Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        "=========================================="
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()