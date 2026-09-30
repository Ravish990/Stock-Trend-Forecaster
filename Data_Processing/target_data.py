import numpy as np
import pandas as pd


# ============================================================
# RSI (Relative Strength Index, 14-day)
# ============================================================

def _compute_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# MACD (12/26 EMA difference + 9 EMA signal)
# ============================================================

def _compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    return macd_line, signal_line


# ============================================================
# Main Data Processing
# ============================================================

def get_data(horizon=1):
    """
    horizon:
        1 -> next trading day
        3 -> 3 trading days ahead
        5 -> 5 trading days ahead

    Main project targets:
        1. next_log_return:
               log(P[t+1] / P[t])
           Used by the regression/price-prediction head.

        2. direction_target:
               1 if next-day close > today's close
               0 otherwise
           Used by the UP/DOWN classification head.

    The raw next_return and next_close are retained so the model's
    predicted log-return can later be converted into predicted price.
    """

    df = pd.read_csv("SP500_Historical_Data_cleaned.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    # ============================================================
    # IMPORTANT: Sort before all rolling / groupby calculations
    # ============================================================

    df = (
        df.sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    # ============================================================
    # TARGETS
    # ============================================================

    # Future close
    df["next_close"] = (
        df.groupby("Ticker")["Close"]
        .shift(-horizon)
    )

    # Simple next return
    df["next_return"] = (
        df["next_close"] - df["Close"]
    ) / df["Close"]

    # Log return:
    #
    #     log(P[t+h] / P[t])
    #
    # This is the regression target.
    df["next_log_return"] = np.log(
        df["next_close"] / df["Close"]
    )

    # UP / DOWN target:
    #
    # 1 -> next close is higher than today's close
    # 0 -> next close is unchanged or lower
    #
    # This is the classification target.
    df["direction_target"] = (
        df["next_return"] > 0
    ).astype(np.float32)

    # ============================================================
    # PER-TICKER FEATURES
    # ============================================================

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

    df["MA5_to_MA20"] = (
        df["MA5"] / df["MA20"] - 1
    )

    # Volatility
    df["volatility_5d"] = (
        df.groupby("Ticker")["return_1d"]
        .transform(lambda x: x.rolling(5).std())
    )

    df["volatility_20d"] = (
        df.groupby("Ticker")["return_1d"]
        .transform(lambda x: x.rolling(20).std())
    )
    df["volume_change"] = (
        df.groupby("Ticker")["Volume"]
        .pct_change()
    )

    avg_volume_20 = (
        df.groupby("Ticker")["Volume"]
        .transform(lambda x: x.rolling(20).mean())
    )

    # Current volume relative to its own recent average
    df["relative_volume_20"] = (
        df["Volume"] / avg_volume_20
    ) - 1

    # Calendar
    month = df["Date"].dt.month

    df["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    # ============================================================
    # RSI
    # ============================================================

    df["RSI_14"] = (
        df.groupby("Ticker")["Close"]
        .transform(lambda x: _compute_rsi(x, 14))
    )

    # Map [0,100] approximately to [-1,1]
    df["RSI_14"] = (
        df["RSI_14"] - 50
    ) / 50

    # ============================================================
    # MACD HISTOGRAM
    # ============================================================

    macd_hist_list = []

    for ticker, group in df.groupby("Ticker"):
        macd_line, signal_line = _compute_macd(
            group["Close"]
        )

        # Normalize by price so the feature is comparable
        # across stocks with different price levels.
        hist = (
            macd_line - signal_line
        ) / group["Close"]

        hist.index = group.index
        macd_hist_list.append(hist)

    df["macd_hist"] = (
        pd.concat(macd_hist_list)
        .sort_index()
    )

    # ============================================================
    # BOLLINGER %B
    # ============================================================

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

    # ============================================================
    # MARKET / CROSS-SECTIONAL FEATURES
    # ============================================================

    # Equal-weighted market proxy from all stocks available
    # on each date.
    market_return_by_date = (
        df.groupby("Date")["return_1d"]
        .mean()
        .sort_index()
    )

    market_return_1d_by_date = market_return_by_date

    # Map daily market return back to every stock row
    df["market_return_1d"] = (
        df["Date"]
        .map(market_return_1d_by_date)
    )

    # Market 5-day cumulative return
    market_return_5d_by_date = (
        (1 + market_return_by_date)
        .rolling(5)
        .apply(np.prod, raw=True)
        - 1
    )

    df["market_return_5d"] = (
        df["Date"]
        .map(market_return_5d_by_date)
    )

    # Cross-sectional excess return of the stock vs the
    # equal-weighted market proxy.
    df["excess_return_1d"] = (
        df["return_1d"]
        - df["market_return_1d"]
    )

    # Cross-sectional market volatility
    market_volatility_by_date = (
        df.groupby("Date")["return_1d"]
        .std()
        .sort_index()
    )

    df["market_volatility"] = (
        df["Date"]
        .map(market_volatility_by_date)
    )

    # Fraction of stocks with positive return on that date.
    market_breadth_by_date = (
        df.groupby("Date")["return_1d"]
        .apply(lambda x: (x > 0).mean())
    )

    df["market_breadth"] = (
        df["Date"]
        .map(market_breadth_by_date)
    )

    # ============================================================
    # MARKET TREND
    # ============================================================

    market_ma20_by_date = (
        market_return_by_date
        .rolling(20)
        .mean()
    )

    df["market_return_1d_vs_ma20"] = (
        df["Date"].map(market_return_1d_by_date)
        - df["Date"].map(market_ma20_by_date)
    )

    # ============================================================
    # ROLLING BETA
    #
    # beta = Cov(stock return, market return)
    #        / Var(market return)
    #
    # Computed using only returns known up to date t.
    # ============================================================

    market_var_20_by_date = (
        market_return_by_date
        .rolling(20)
        .var()
    )

    market_var_60_by_date = (
        market_return_by_date
        .rolling(60)
        .var()
    )

    df["market_var_20"] = (
        df["Date"]
        .map(market_var_20_by_date)
    )

    df["market_var_60"] = (
        df["Date"]
        .map(market_var_60_by_date)
    )

    beta20_list = []
    beta60_list = []

    for ticker, group in df.groupby("Ticker"):
        stock_return = group["return_1d"]
        market_return = group["market_return_1d"]

        cov20 = (
            stock_return
            .rolling(20)
            .cov(market_return)
        )

        cov60 = (
            stock_return
            .rolling(60)
            .cov(market_return)
        )

        beta20 = (
            cov20 / group["market_var_20"]
        )

        beta60 = (
            cov60 / group["market_var_60"]
        )

        beta20.index = group.index
        beta60.index = group.index

        beta20_list.append(beta20)
        beta60_list.append(beta60)

    df["rolling_beta_20"] = (
        pd.concat(beta20_list)
        .sort_index()
    )

    df["rolling_beta_60"] = (
        pd.concat(beta60_list)
        .sort_index()
    )

    # ============================================================
    # RESIDUAL / IDIOSYNCRATIC RETURN
    #
    # residual = stock return - beta * market return
    #
    # This gives the model a market-adjusted momentum signal
    # without using future information.
    # ============================================================

    df["residual_return_1d"] = (
        df["return_1d"]
        - df["rolling_beta_20"]
        * df["market_return_1d"]
    )

    df["residual_return_5d"] = (
        df.groupby("Ticker")["residual_return_1d"]
        .transform(
            lambda x: x.rolling(5).sum()
        )
    )

    # ============================================================
    # CLEANUP
    # ============================================================

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    required_columns = [
        # Targets
        "next_close",
        "next_return",
        "next_log_return",
        "direction_target",

        # Momentum / trend
        "return_1d",
        "return_5d",
        "return_20d",
        "price_to_MA5",
        "price_to_MA20",
        "MA5_to_MA20",
        "volatility_5d",
        "volatility_20d",
        "volume_change",
        "relative_volume_20",

        # Technical indicators
        "RSI_14",
        "macd_hist",
        "bollinger_pct_b",

        # Market-relative
        "market_return_1d",
        "market_return_5d",
        "excess_return_1d",
        "market_volatility",
        "market_breadth",
        "market_return_1d_vs_ma20",

        # Beta / residual
        "rolling_beta_20",
        "rolling_beta_60",
        "residual_return_1d",
        "residual_return_5d",
    ]

    df = df.dropna(
        subset=required_columns
    ).copy()

    return df


# ============================================================
# Example
# ============================================================

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
