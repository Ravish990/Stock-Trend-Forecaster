
import pandas as pd
from target_data import get_data

NEWS_PATH = "all_ticker_news_daily_2020_2026.csv"
OUTPUT_PATH = "merged_news_target_data.csv"
HORIZON = 5

sentiment_columns = [
    "News_Count",
    "Avg_Sentiment",
    "Avg_Relevance",
    "Sentiment_Std",
    "Max_Sentiment",
    "Min_Sentiment",
    "Weighted_Sentiment",
    "Positive_Count",
    "Negative_Count",
    "Neutral_Count",
]

# Load news
news_df = pd.read_csv(NEWS_PATH)
news_df["Date"] = pd.to_datetime(
    news_df["Date"], errors="coerce"
)
news_df = news_df.dropna(subset=["Date", "Ticker"]).copy()
news_df["Ticker"] = news_df["Ticker"].astype(str).str.strip()

# Load target/feature data
# Keep this horizon consistent with train_model.py.
target_df = get_data(horizon=HORIZON)
target_df["Date"] = pd.to_datetime(
    target_df["Date"], errors="coerce"
)
target_df = target_df.dropna(subset=["Date", "Ticker"]).copy()
target_df["Ticker"] = target_df["Ticker"].astype(str).str.strip()

# Keep available sentiment columns only
available_sentiments = [
    col for col in sentiment_columns if col in news_df.columns
]

if not available_sentiments:
    raise ValueError("No expected sentiment columns found in news CSV.")

news_df = news_df[
    ["Ticker", "Date"] + available_sentiments
].copy()

# Ensure sentiment values are numeric
for col in available_sentiments:
    news_df[col] = pd.to_numeric(news_df[col], errors="coerce")

# Aggregate daily sentiment.
# Counts are summed; average/summary statistics are averaged
# if duplicate ticker-date rows exist.
aggregation = {}

for col in available_sentiments:
    if col in ["News_Count", "Positive_Count",
               "Negative_Count", "Neutral_Count"]:
        aggregation[col] = "sum"
    else:
        aggregation[col] = "mean"

news_df = news_df.groupby(
    ["Ticker", "Date"], as_index=False
).agg(aggregation)

# Validate target uniqueness before merge
if target_df.duplicated(["Ticker", "Date"]).any():
    raise ValueError(
        "Target data has duplicate Ticker + Date rows. "
        "Fix duplicates before merging."
    )

# Left merge: preserve target rows
df = target_df.merge(
    news_df,
    how="left",
    on=["Ticker", "Date"],
    validate="one_to_one",
    indicator=True,
)

matched_rows = int((df["_merge"] == "both").sum())
unmatched_rows = int((df["_merge"] == "left_only").sum())
df = df.drop(columns=["_merge"])

# No matching news means no recorded news for that ticker/date.
# Fill missing sentiment features with zero.
for col in sentiment_columns:
    if col not in df.columns:
        df[col] = 0
    else:
        df[col] = df[col].fillna(0)

print("Target rows:", len(target_df))
print("Merged rows:", len(df))
print("Matched ticker-date rows:", matched_rows)
print("Unmatched ticker-date rows:", unmatched_rows)
print("Target date range:",
      target_df["Date"].min(), "to", target_df["Date"].max())
print("Merged date range:",
      df["Date"].min(), "to", df["Date"].max())

assert len(df) == len(target_df), (
    "Merge changed row count; investigate duplicate keys."
)

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
print("Final shape:", df.shape)
