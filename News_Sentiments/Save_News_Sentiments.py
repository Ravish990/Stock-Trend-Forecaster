import requests
import pandas as pd
import time


# ============================================================
# CONFIG
# ============================================================

API_KEY = "ADUNOG7XCWSYT5H1"

START_DATE = "2020-01-01"
END_DATE = "2026-10-06"

BASE_URL = "https://www.alphavantage.co/query"

OUTPUT_RAW = "all_news_sentiment_raw_2020_2026.csv"
OUTPUT_DAILY = "all_ticker_news_daily_2020_2026.csv"


# ============================================================
# FETCH NEWS
# ============================================================

def fetch_news(start_date, end_date):

    time_from = pd.Timestamp(start_date).strftime(
        "%Y%m%dT0000"
    )

    time_to = pd.Timestamp(end_date).strftime(
        "%Y%m%dT2359"
    )

    params = {
        "function": "NEWS_SENTIMENT",
        "time_from": time_from,
        "time_to": time_to,
        "limit": 1000,
        "sort": "EARLIEST",
        "apikey": API_KEY
    }

    print(f"\nDownloading:")
    print(f"{start_date} -> {end_date}")

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "feed" not in data:

        print("\nAPI did not return feed:")
        print(data)

        return []

    return data["feed"]


# ============================================================
# EXTRACT ALL TICKERS FROM EACH ARTICLE
# ============================================================

def extract_all_tickers(feed):

    rows = []

    for article in feed:

        published = article.get("time_published")

        if not published:
            continue

        published_dt = pd.to_datetime(
            published,
            format="%Y%m%dT%H%M%S"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Loop through EVERY ticker in ticker_sentiment
        # ----------------------------------------------------

        for ticker_data in article.get(
            "ticker_sentiment",
            []
        ):

            ticker = ticker_data.get("ticker")

            if not ticker:
                continue

            rows.append({

                # Article information
                "Ticker": ticker,

                "Date": published_dt.date(),

                "Published_Time": published_dt,

                "Title": article.get("title"),

                "Summary": article.get("summary"),

                "Source": article.get("source"),

                "Source_Domain":
                    article.get("source_domain"),

                "URL": article.get("url"),

                # Article-level sentiment
                "Overall_Sentiment":
                    article.get(
                        "overall_sentiment_score"
                    ),

                "Overall_Label":
                    article.get(
                        "overall_sentiment_label"
                    ),

                # Ticker-specific sentiment
                "Ticker_Relevance":
                    ticker_data.get(
                        "relevance_score"
                    ),

                "Ticker_Sentiment":
                    ticker_data.get(
                        "ticker_sentiment_score"
                    ),

                "Ticker_Sentiment_Label":
                    ticker_data.get(
                        "ticker_sentiment_label"
                    )
            })

    return rows


# ============================================================
# CREATE 6-MONTH WINDOWS
# ============================================================

def create_windows(start_date, end_date):

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    windows = []

    current = start

    while current <= end:

        next_date = (
            current
            + pd.DateOffset(months=6)
            - pd.Timedelta(days=1)
        )

        if next_date > end:
            next_date = end

        windows.append((
            current.strftime("%Y-%m-%d"),
            next_date.strftime("%Y-%m-%d")
        ))

        current = (
            next_date
            + pd.Timedelta(days=1)
        )

    return windows


# ============================================================
# MAIN
# ============================================================

def main():

    all_rows = []

    windows = create_windows(
        START_DATE,
        END_DATE
    )

    print(
        f"\nTotal date windows: {len(windows)}"
    )

    # --------------------------------------------------------
    # Download every window
    # --------------------------------------------------------

    for i, (start, end) in enumerate(
        windows,
        start=1
    ):

        print("\n" + "=" * 70)

        print(
            f"WINDOW {i}/{len(windows)}"
        )

        feed = fetch_news(
            start,
            end
        )

        print(
            f"Articles received: {len(feed)}"
        )

        rows = extract_all_tickers(feed)

        print(
            f"Ticker sentiment rows: {len(rows)}"
        )

        all_rows.extend(rows)

        # Avoid hammering API
        time.sleep(1)


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if not all_rows:

        print("\nNo data collected.")
        return

    df = pd.DataFrame(all_rows)


    # ========================================================
    # CLEAN DATA
    # ========================================================

    df["Ticker_Relevance"] = pd.to_numeric(
        df["Ticker_Relevance"],
        errors="coerce"
    )

    df["Ticker_Sentiment"] = pd.to_numeric(
        df["Ticker_Sentiment"],
        errors="coerce"
    )

    df["Overall_Sentiment"] = pd.to_numeric(
        df["Overall_Sentiment"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Remove duplicate article/ticker combinations
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "URL",
            "Ticker"
        ]
    )


    # Sort
    df = df.sort_values(
        [
            "Ticker",
            "Published_Time"
        ]
    )


    # ========================================================
    # SAVE RAW TICKER SENTIMENT
    # ========================================================

    df.to_csv(
        OUTPUT_RAW,
        index=False
    )

    print("\n" + "=" * 70)

    print("RAW DATA SAVED")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Tickers: {df['Ticker'].nunique()}"
    )

    print(
        f"File: {OUTPUT_RAW}"
    )


    # ========================================================
    # DAILY AGGREGATION
    # ========================================================

    daily = (

        df.groupby(
            [
                "Ticker",
                "Date"
            ],
            as_index=False
        )

        .agg(

            News_Count=(
                "URL",
                "count"
            ),

            Avg_Sentiment=(
                "Ticker_Sentiment",
                "mean"
            ),

            Avg_Relevance=(
                "Ticker_Relevance",
                "mean"
            ),

            Sentiment_Std=(
                "Ticker_Sentiment",
                "std"
            ),

            Max_Sentiment=(
                "Ticker_Sentiment",
                "max"
            ),

            Min_Sentiment=(
                "Ticker_Sentiment",
                "min"
            )
        )
    )


    # ========================================================
    # WEIGHTED SENTIMENT
    # ========================================================

    df["Weighted_Value"] = (
        df["Ticker_Sentiment"]
        * df["Ticker_Relevance"]
    )

    weighted = (

        df.groupby(
            [
                "Ticker",
                "Date"
            ]
        )

        .agg(
            Weighted_Sum=(
                "Weighted_Value",
                "sum"
            ),

            Relevance_Sum=(
                "Ticker_Relevance",
                "sum"
            )
        )

        .reset_index()
    )


    weighted["Weighted_Sentiment"] = (
        weighted["Weighted_Sum"]
        / weighted["Relevance_Sum"]
    )


    daily = daily.merge(
        weighted[
            [
                "Ticker",
                "Date",
                "Weighted_Sentiment"
            ]
        ],
        on=[
            "Ticker",
            "Date"
        ],
        how="left"
    )


    # ========================================================
    # POSITIVE / NEGATIVE / NEUTRAL COUNTS
    # ========================================================

    df["Positive"] = (
        df["Ticker_Sentiment"] >= 0.15
    )

    df["Negative"] = (
        df["Ticker_Sentiment"] <= -0.15
    )

    df["Neutral"] = (
        (df["Ticker_Sentiment"] > -0.15)
        &
        (df["Ticker_Sentiment"] < 0.15)
    )


    sentiment_counts = (

        df.groupby(
            [
                "Ticker",
                "Date"
            ]
        )

        .agg(

            Positive_Count=(
                "Positive",
                "sum"
            ),

            Negative_Count=(
                "Negative",
                "sum"
            ),

            Neutral_Count=(
                "Neutral",
                "sum"
            )
        )

        .reset_index()
    )


    daily = daily.merge(
        sentiment_counts,
        on=[
            "Ticker",
            "Date"
        ],
        how="left"
    )


    # ========================================================
    # SAVE DAILY DATA
    # ========================================================

    daily = daily.sort_values(
        [
            "Ticker",
            "Date"
        ]
    )


    daily.to_csv(
        OUTPUT_DAILY,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print("DAILY DATA SAVED")

    print(
        f"Rows: {len(daily):,}"
    )

    print(
        f"Unique tickers: "
        f"{daily['Ticker'].nunique()}"
    )

    print(
        f"Date range: "
        f"{daily['Date'].min()} -> "
        f"{daily['Date'].max()}"
    )

    print(
        f"File: {OUTPUT_DAILY}"
    )

    print("\nTickers found:")

    print(
        daily["Ticker"]
        .value_counts()
        .head(30)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()