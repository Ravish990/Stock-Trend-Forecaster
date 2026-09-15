import pandas as pd
import numpy as np

df = pd.read_csv('../SP500_Historical_Data_cleaned.csv')

# Target
df["next_close"] = df.groupby("Ticker")["Close"].shift(-1)

df["next_return"] = (
    df["next_close"] - df["Close"]
) / df["Close"]

df["target"] = (df["next_return"] > 0).astype(int)


# =========================
# Feature Engineering
# =========================

# 1. 1-day return
df["return_1d"] = (
    df.groupby("Ticker")["Close"].pct_change()
)

# 2. 5-day return
df["return_5d"] = (
    df.groupby("Ticker")["Close"].pct_change(5)
)

# 3. Moving Average 5 days
df["MA5"] = (
    df.groupby("Ticker")["Close"]
      .transform(lambda x: x.rolling(5).mean())
)

# 4. Moving Average 20 days
df["MA20"] = (
    df.groupby("Ticker")["Close"]
      .transform(lambda x: x.rolling(20).mean())
)

# 5. 5-day volatility
df["volatility_5d"] = (
    df.groupby("Ticker")["return_1d"]
      .transform(lambda x: x.rolling(5).std())
)

# 6. Volume change
df["volume_change"] = (
    df.groupby("Ticker")["Volume"].pct_change()
)


# =========================
# Seasonality Features
# =========================

df["Date"] = pd.to_datetime(df["Date"])

# Month
df["month"] = df["Date"].dt.month

# Cyclical representation of month
df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

df = df.dropna(subset=[
        "next_close",
        "next_return",
        "target",
        "return_1d",
        "return_5d",
        "MA5",
        "MA20",
        "volatility_5d",
        "volume_change",
    ]).copy()




print(df.head(30))