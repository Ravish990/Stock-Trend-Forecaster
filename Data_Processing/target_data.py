import numpy as np
import pandas as pd

def _compute_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def _compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    return macd_line, signal_line

def get_data(horizon=1):
    df = pd.read_csv("SP500_Historical_Data_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    # Future close
    df["next_close"] = (
        df.groupby("Ticker")["Close"]
        .shift(-horizon)
    )

    df["next_date"] = (
        df.groupby("Ticker")["Date"]
        .shift(-horizon)
    )

    # Simple next return
    df["next_return"] = (
        df["next_close"] - df["Close"]
    ) / df["Close"]
    
    df["next_log_return"] = np.log(
        df["next_close"] / df["Close"]
    )
    
    df["direction_target"] = np.where(
        df["next_close"].notna(),
        (df["next_return"] > 0).astype(np.float32),
        np.nan,
    )

    df["return_1d"] = (
        df.groupby("Ticker")["Close"]
        .pct_change()
    )

    df["return_5d"] = (
        df.groupby("Ticker")["Close"]
        .pct_change(5)
    )

    df["return_20d"] = (
        df.groupby("Ticker")["Close"]
        .pct_change(20)
    )

    # --- NEW: Momentum Features (12-1 and 6-1) ---
    # 12-month momentum skipping the most recent month (252 days - 21 days)
    df["mom_12_1"] = (
        df.groupby("Ticker")["Close"].shift(21) / df.groupby("Ticker")["Close"].shift(252)
    ) - 1
    
    # 6-month momentum skipping the most recent month (126 days - 21 days)
    df["mom_6_1"] = (
        df.groupby("Ticker")["Close"].shift(21) / df.groupby("Ticker")["Close"].shift(126)
    ) - 1

    # Moving averages
    df["MA5"] = (
        df.groupby("Ticker")["Close"]
        .transform(lambda x: x.rolling(5).mean())
    )

    df["MA20"] = (
        df.groupby("Ticker")["Close"]
        .transform(lambda x: x.rolling(20).mean())
    )

    df["price_to_MA5"] = (
        df["Close"] / df["MA5"] - 1
    )

    df["price_to_MA20"] = (
        df["Close"] / df["MA20"] - 1
    )

    # Volatility
    df["volatility_5d"] = (
        df.groupby("Ticker")["return_1d"]
        .transform(lambda x: x.rolling(5).std())
    )
    
    # --- NEW: 60-day Volatility ---
    df["volatility_60d"] = (
        df.groupby("Ticker")["return_1d"]
        .transform(lambda x: x.rolling(60).std())
    )

    avg_volume_20 = (
        df.groupby("Ticker")["Volume"]
        .transform(lambda x: x.rolling(20).mean())
    )

    df["relative_volume_20"] = (
        df["Volume"] / avg_volume_20
    ) - 1

    month = df["Date"].dt.month

    df["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    df["RSI_14"] = (
        df.groupby("Ticker")["Close"]
        .transform(lambda x: _compute_rsi(x, 14))
    )

    df["RSI_14"] = (
        df["RSI_14"] - 50
    ) / 50

    rolling_mean_20 = (
        df.groupby("Ticker")["Close"]
        .transform(lambda x: x.rolling(20).mean())
    )

    rolling_std_20 = (
        df.groupby("Ticker")["Close"]
        .transform(lambda x: x.rolling(20).std())
    )

    upper_band = (
        rolling_mean_20 + 2 * rolling_std_20
    )

    lower_band = (
        rolling_mean_20 - 2 * rolling_std_20
    )

    band_width = upper_band - lower_band

    df["bollinger_pct_b"] = (
        (df["Close"] - lower_band)
        / band_width
    )

    market_return_by_date = (
        df.groupby("Date")["return_1d"]
        .mean()
        .sort_index()
    )
    
    # Need to map this to the dataframe so we can calculate excess and beta
    df["market_return_1d"] = df["Date"].map(market_return_by_date)

    df["excess_return_1d"] = (
        df["return_1d"]
        - df["market_return_1d"]
    )

    market_var_20_by_date = (
        market_return_by_date
        .rolling(20)
        .var()
    )

    df["market_var_20"] = (
        df["Date"]
        .map(market_var_20_by_date)
    )

    beta20_list = []

    for ticker, group in df.groupby("Ticker"):
        stock_return = group["return_1d"]
        market_return = group["market_return_1d"]

        cov20 = (
            stock_return
            .rolling(20)
            .cov(market_return)
        )

        beta20 = (
            cov20 / group["market_var_20"]
        )

        beta20.index = group.index

        beta20_list.append(beta20)

    df["rolling_beta_20"] = (
        pd.concat(beta20_list)
        .sort_index()
        .clip(-5, 5)  # clip extreme outliers from near-zero market variance days
    )

    # ATR-14 normalized by price
    df["high_low"] = df["High"] - df["Low"]
    df["atr_14"] = (
        df.groupby("Ticker")["high_low"]
        .transform(lambda x: x.rolling(14).mean())
    ) / df["Close"]

    # Distance from 52-week high/low
    df["dist_52w_high"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x / x.rolling(252).max() - 1
    )
    df["dist_52w_low"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x / x.rolling(252).min() - 1
    )

    df["residual_return_1d"] = (
        df["return_1d"]
        - df["rolling_beta_20"]
        * df["market_return_1d"]
    )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    required_columns = [
        # Targets
        "next_close",
        "next_date",
        "next_return",
        "next_log_return",
        "direction_target",

        # Momentum / trend
        "return_1d",
        "return_5d",
        "return_20d",
        "mom_12_1",       # NEW
        "mom_6_1",        # NEW
        "price_to_MA5",
        "price_to_MA20",
        "volatility_5d",
        "volatility_60d", # NEW
        "relative_volume_20",

        # Technical indicators
        "RSI_14",
        "bollinger_pct_b",

        # Market-relative
        "excess_return_1d",

        # Beta / residual
        "rolling_beta_20",
        "residual_return_1d",

        # Price structure
        "atr_14",
        "dist_52w_high",
        "dist_52w_low",
    ]

    df = df.dropna(
        subset=required_columns
    ).copy()

    return df

if __name__ == "__main__":
    data = get_data(horizon=1)

    print("\n===== DATA =====")
    print(data.shape)

    print("\n===== TARGETS =====")
    print(
        data[
            [
                "Date",
                "Ticker",
                "Close",
                "next_close",
                "next_return",
                "next_log_return",
                "direction_target",
            ]
        ].head(10)
    )

    print("\n===== FEATURES =====")
    print(data.columns.tolist())