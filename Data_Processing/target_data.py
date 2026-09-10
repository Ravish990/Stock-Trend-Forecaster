import pandas as pd

df = pd.read_csv('../SP500_Historical_Data_cleaned.csv')
df["next_close"] = df.groupby("Ticker")["Close"].shift(-1)

df["next_return"] = (
    df["next_close"] - df["Close"]
) / df["Close"]

df["target"] = (df["next_return"] > 0).astype(int)

print("Target variable created. Here are the first few rows:")
print(df.head())