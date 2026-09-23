import numpy as np
import pandas as pd


def get_data():

    df = pd.read_csv('SP500_Historical_Data_cleaned.csv')

    # Target
    df["next_close"] = df.groupby("Ticker")["Close"].shift(-1)

    df["next_return"] = (
        df["next_close"] - df["Close"]
    ) / df["Close"]

    df["target"] = (df["next_return"] > 0).astype(int)

    # Features
    df["return_1d"] = df.groupby("Ticker")["Close"].pct_change()

    df["return_5d"] = df.groupby("Ticker")["Close"].pct_change(5)

    df["MA5"] = (
        df.groupby("Ticker")["Close"]
        .transform(lambda x: x.rolling(5).mean())
    )

    df["MA20"] = (
        df.groupby("Ticker")["Close"]
        .transform(lambda x: x.rolling(20).mean())
    )

    # --- Relative (stationary) trend features, not raw price levels ---
    df["price_to_MA5"] = df["Close"] / df["MA5"] - 1
    df["price_to_MA20"] = df["Close"] / df["MA20"] - 1
    df["MA5_to_MA20"] = df["MA5"] / df["MA20"] - 1

    df["volatility_5d"] = (
        df.groupby("Ticker")["return_1d"]
        .transform(lambda x: x.rolling(5).std())
    )

    df["volume_change"] = (
        df.groupby("Ticker")["Volume"].pct_change()
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df["month"] = df["Date"].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.dropna(subset=[
        "next_close",
        "next_return",
        "target",
        "return_1d",
        "return_5d",
        "price_to_MA5",
        "price_to_MA20",
        "MA5_to_MA20",
        "volatility_5d",
        "volume_change",
    ]).copy()

    return df