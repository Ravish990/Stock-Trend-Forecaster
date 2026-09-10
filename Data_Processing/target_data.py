import pandas as pd

df = pd.read_csv('../SP500_Historical_Data_cleaned.csv')
df["next_close"] = df.groupby("Ticker")["Close"].shift(-1)

df["next_return"] = (
    df["next_close"] - df["Close"]
) / df["Close"]

df["target"] = (df["next_return"] > 0).astype(int)

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

df["volatility_5d"] = (
    df.groupby("Ticker")["return_1d"]
    .transform(lambda x: x.rolling(5).std())
)

df["volume_change"] = (
    df.groupby("Ticker")["Volume"].pct_change()
)

print("Target variable created. Here are the first few rows:")
print(df.head(30))