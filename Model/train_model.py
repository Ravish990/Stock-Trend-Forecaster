import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from Data_Processing.target_data import get_data
from Model.model import LSTMModel


# ============================================================
# Create Sequences With Date Filtering
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

    for ticker, group in df.groupby("Ticker"):

        group = group.sort_values("Date").reset_index(drop=True)

        X = group[features].values

        prices = group["next_close"].values
        directions = group["target"].values
        target_dates = group["next_date"].values

        for i in range(sequence_length, len(group)):

            target_date = target_dates[i]

            # -----------------------------------------------
            # Filter target dates
            # -----------------------------------------------

            if min_target_date is not None:
                if target_date < np.datetime64(min_target_date):
                    continue

            if max_target_date is not None:
                if target_date >= np.datetime64(max_target_date):
                    continue

            # -----------------------------------------------
            # Create sequence
            # -----------------------------------------------

            X_sequences.append(
                X[i-sequence_length:i]
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

            # -----------------------------------------------
            # Prediction
            # -----------------------------------------------

            price_pred, direction_pred = model(
                X_batch
            )

            price_pred = price_pred.squeeze(1)
            direction_pred = direction_pred.squeeze(1)

            # -----------------------------------------------
            # Price RMSE
            # -----------------------------------------------

            squared_error = (
                price_pred - price_batch
            ) ** 2

            total_squared_error += (
                squared_error.sum().item()
            )

            # -----------------------------------------------
            # Direction Accuracy
            # -----------------------------------------------

            direction_probability = torch.sigmoid(
                direction_pred
            )

            direction_prediction = (
                direction_probability >= 0.5
            ).float()

            total_correct += (
                direction_prediction == direction_batch
            ).sum().item()

            total_samples += direction_batch.size(0)

    # -----------------------------------------------
    # Metrics
    # -----------------------------------------------

    rmse = np.sqrt(
        total_squared_error / total_samples
    )

    directional_accuracy = (
        total_correct / total_samples
    ) * 100

    return rmse, directional_accuracy


# ============================================================
# Training
# ============================================================

def train_model():

    # ========================================================
    # Load Already-Cleaned Data
    # ========================================================

    df = get_data()

    # Date conversion
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort data
    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    # ========================================================
    # Create Next Target Date
    # ========================================================

    df["next_date"] = (
        df.groupby("Ticker")["Date"]
        .shift(-1)
    )

    # ========================================================
    # Features
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
    # Split Dates
    # ========================================================

    validation_start = pd.Timestamp("2012-01-01")
    test_start = pd.Timestamp("2015-01-01")

    # ========================================================
    # Training Data
    # ========================================================

    train_rows = df[
        df["Date"] < validation_start
    ].copy()

    # ========================================================
    # Feature Scaling
    # ========================================================
    #
    # Fit ONLY on training data
    # ========================================================

    scaler = StandardScaler()

    scaler.fit(
        train_rows[features]
    )

    # Apply the training scaler to all data

    df[features] = scaler.transform(
        df[features]
    )

    # ========================================================
    # Training Sequences
    # ========================================================

    X_train, price_train, direction_train = (
        create_sequences_with_date_filter(
            df,
            features,
            sequence_length=20,
            min_target_date=None,
            max_target_date=validation_start
        )
    )

    # ========================================================
    # Validation Context
    # ========================================================
    #
    # Last 20 observations before validation period
    # are used as input context.
    # ========================================================

    validation_context = (
        df[
            df["Date"] < validation_start
        ]
        .groupby("Ticker")
        .tail(20)
    )

    validation_data = df[
        (df["Date"] >= validation_start) &
        (df["Date"] < test_start)
    ]

    validation_sequence_df = pd.concat(
        [
            validation_context,
            validation_data
        ]
    )

    validation_sequence_df = (
        validation_sequence_df
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    X_val, price_val, direction_val = (
        create_sequences_with_date_filter(
            validation_sequence_df,
            features,
            sequence_length=20,
            min_target_date=validation_start,
            max_target_date=test_start
        )
    )

    # ========================================================
    # Test Context
    # ========================================================

    test_context = (
        df[
            df["Date"] < test_start
        ]
        .groupby("Ticker")
        .tail(20)
    )

    test_data = df[
        df["Date"] >= test_start
    ]

    test_sequence_df = pd.concat(
        [
            test_context,
            test_data
        ]
    )

    test_sequence_df = (
        test_sequence_df
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
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
    # Print Shapes
    # ========================================================

    print("\n===== DATA SHAPES =====")

    print("X_train:", X_train.shape)
    print("Price train:", price_train.shape)
    print("Direction train:", direction_train.shape)

    print("X_val:", X_val.shape)
    print("Price val:", price_val.shape)
    print("Direction val:", direction_val.shape)

    print("X_test:", X_test.shape)
    print("Price test:", price_test.shape)
    print("Direction test:", direction_test.shape)

    # ========================================================
    # Convert To PyTorch
    # ========================================================

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    price_train = torch.tensor(
        price_train,
        dtype=torch.float32
    )

    direction_train = torch.tensor(
        direction_train,
        dtype=torch.float32
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    price_val = torch.tensor(
        price_val,
        dtype=torch.float32
    )

    direction_val = torch.tensor(
        direction_val,
        dtype=torch.float32
    )

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
    # Dataset
    # ========================================================

    train_dataset = TensorDataset(
        X_train,
        price_train,
        direction_train
    )

    val_dataset = TensorDataset(
        X_val,
        price_val,
        direction_val
    )

    test_dataset = TensorDataset(
        X_test,
        price_test,
        direction_test
    )

    # ========================================================
    # DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # ========================================================
    # Model
    # ========================================================

    model = LSTMModel(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )

    # ========================================================
    # Loss Functions
    # ========================================================

    price_loss_fn = torch.nn.MSELoss()

    direction_loss_fn = (
        torch.nn.BCEWithLogitsLoss()
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # ========================================================
    # Training Settings
    # ========================================================

    epochs = 50

    alpha = 0.01
    beta = 1.0

    # Early stopping
    patience = 5
    best_val_loss = float("inf")
    patience_counter = 0

    model_path = "best_lstm_stock_model.pth"

    # ========================================================
    # Training Loop
    # ========================================================

    for epoch in range(epochs):

        model.train()

        total_train_loss = 0.0

        for (
            X_batch,
            price_batch,
            direction_batch
        ) in train_loader:

            # -----------------------------------------------
            # Forward Pass
            # -----------------------------------------------

            price_pred, direction_pred = model(
                X_batch
            )

            price_pred = price_pred.squeeze(1)
            direction_pred = direction_pred.squeeze(1)

            # -----------------------------------------------
            # Loss
            # -----------------------------------------------

            price_loss = price_loss_fn(
                price_pred,
                price_batch
            )

            direction_loss = direction_loss_fn(
                direction_pred,
                direction_batch
            )

            loss = (
                alpha * price_loss
                + beta * direction_loss
            )

            # -----------------------------------------------
            # Backpropagation
            # -----------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        # ====================================================
        # Validation Loss
        # ====================================================

        model.eval()

        total_val_loss = 0.0

        with torch.no_grad():

            for (
                X_batch,
                price_batch,
                direction_batch
            ) in val_loader:

                price_pred, direction_pred = model(
                    X_batch
                )

                price_pred = price_pred.squeeze(1)
                direction_pred = direction_pred.squeeze(1)

                price_loss = price_loss_fn(
                    price_pred,
                    price_batch
                )

                direction_loss = direction_loss_fn(
                    direction_pred,
                    direction_batch
                )

                val_loss = (
                    alpha * price_loss
                    + beta * direction_loss
                )

                total_val_loss += (
                    val_loss.item()
                )

        average_val_loss = (
            total_val_loss / len(val_loader)
        )

        # ====================================================
        # Validation Metrics
        # ====================================================

        val_rmse, val_accuracy = evaluate_model(
            model,
            val_loader
        )

        # ====================================================
        # Print Results
        # ====================================================

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {average_train_loss:.4f} "
            f"Val Loss: {average_val_loss:.4f} "
            f"Val RMSE: {val_rmse:.4f} "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        # ====================================================
        # Save Best Model
        # ====================================================

        if average_val_loss < best_val_loss:

            best_val_loss = average_val_loss

            patience_counter = 0

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                "  -> Best model saved."
            )

        else:

            patience_counter += 1

            print(
                f"  -> No improvement "
                f"({patience_counter}/{patience})"
            )

        # ====================================================
        # Early Stopping
        # ====================================================

        if patience_counter >= patience:

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # Load Best Model
    # ========================================================

    print(
        "\nLoading best model..."
    )

    model.load_state_dict(
        torch.load(
            model_path,
            weights_only=True
        )
    )

    # ========================================================
    # Final Validation Results
    # ========================================================

    val_rmse, val_accuracy = evaluate_model(
        model,
        val_loader
    )

    # ========================================================
    # Final Test Results
    # ========================================================

    test_rmse, test_accuracy = evaluate_model(
        model,
        test_loader
    )

    # ========================================================
    # Results
    # ========================================================

    print("\n===== VALIDATION RESULTS =====")

    print(
        f"Validation RMSE: {val_rmse:.4f}"
    )

    print(
        f"Validation Directional Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print("\n===== TEST RESULTS =====")

    print(
        f"Test Price RMSE: {test_rmse:.4f}"
    )

    print(
        f"Test Directional Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"\nBest model saved to: {model_path}"
    )

    return model


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    model = train_model()