import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from Data_Processing.target_data import get_data
from Model.model import LSTMModel


def create_sequences(df, features, sequence_length=20):

    X_sequences = []
    price_targets = []
    direction_targets = []

    for ticker, group in df.groupby("Ticker"):

        group = group.sort_values("Date")

        X = group[features].values

        prices = group["next_close"].values
        directions = group["target"].values

        for i in range(sequence_length, len(group)):

            # Previous 20 days
            X_sequences.append(
                X[i-sequence_length:i]
            )

            # Target for the next day
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


def train_model():

    df = get_data()

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

    # Make sure data is sorted
    df = df.sort_values(["Ticker", "Date"])

    # =========================
    # Train-Test Split
    # =========================

    split_date = pd.Timestamp("2002-01-01")

    train_df = df[df["Date"] < split_date].copy()
    test_df = df[df["Date"] >= split_date].copy()

    # =========================
    # Create Sequences
    # =========================

    X_train, price_train, direction_train = create_sequences(
        train_df,
        features,
        sequence_length=20
    )

    X_test, price_test, direction_test = create_sequences(
        test_df,
        features,
        sequence_length=20
    )

    print("X_train:", X_train.shape)
    print("Price train:", price_train.shape)
    print("Direction train:", direction_train.shape)

    print("X_test:", X_test.shape)
    print("Price test:", price_test.shape)
    print("Direction test:", direction_test.shape)

    # =========================
    # Convert to PyTorch tensors
    # =========================

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

    # =========================
    # Dataset
    # =========================

    train_dataset = TensorDataset(
        X_train,
        price_train,
        direction_train
    )

    test_dataset = TensorDataset(
        X_test,
        price_test,
        direction_test
    )

    # =========================
    # DataLoader
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # =========================
    # Create Model
    # =========================

    model = LSTMModel(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )

    # =========================
    # Loss Functions
    # =========================

    price_loss_fn = torch.nn.MSELoss()

    direction_loss_fn = torch.nn.BCEWithLogitsLoss()

    # =========================
    # Optimizer
    # =========================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # =========================
    # Training
    # =========================

    epochs = 10

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for X_batch, price_batch, direction_batch in train_loader:

            # Forward pass
            price_pred, direction_pred = model(X_batch)

            # Remove extra dimension
            price_pred = price_pred.squeeze(1)
            direction_pred = direction_pred.squeeze(1)

            # Calculate losses
            price_loss = price_loss_fn(
                price_pred,
                price_batch
            )

            direction_loss = direction_loss_fn(
                direction_pred,
                direction_batch
            )

            # Combined loss
            loss = price_loss + direction_loss

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {average_loss:.4f}"
        )

    return model

if __name__ == "__main__":
    model = train_model()