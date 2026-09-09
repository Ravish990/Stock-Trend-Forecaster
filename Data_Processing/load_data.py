import pandas as pd

df = pd.read_csv('../SP500_Historical_Data.csv')

print("Original shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

df.columns = df.columns.str.strip()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

numeric_columns = [
    'Open',
    'High',
    'Low',
    'Close',
    'Adj Close',
    'Volume'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.dropna().copy()

duplicates = df.duplicated().sum()
print("\nExact duplicate rows:", duplicates)

df = df.drop_duplicates().copy()

ticker_date_duplicates = df.duplicated(
    subset=['Ticker', 'Date']
).sum()

print("Duplicate Ticker + Date rows:", ticker_date_duplicates)

df = df.drop_duplicates(
    subset=['Ticker', 'Date'],
    keep='first'
).copy()

invalid_ohlc = (
    (df['High'] < df['Open']) |
    (df['High'] < df['Close']) |
    (df['Low'] > df['Open']) |
    (df['Low'] > df['Close']) |
    (df['High'] < df['Low'])
)

print("\nInvalid OHLC rows:", invalid_ohlc.sum())

df = df[~invalid_ohlc].copy()

negative_volume = (df['Volume'] < 0).sum()

print("Negative volume rows:", negative_volume)

df = df[df['Volume'] >= 0].copy()

df = df.dropna(subset=['Date']).copy()

df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()

df = df[df['Ticker'] != ''].copy()

df = df.sort_values(
    ['Ticker', 'Date']
).reset_index(drop=True)

print("\n=========================")
print("FINAL DATASET")
print("=========================")

print("Shape:", df.shape)

print("Unique tickers:", df['Ticker'].nunique())

print("Date range:")
print(df['Date'].min(), "to", df['Date'].max())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDuplicate Ticker + Date:")
print(df.duplicated(subset=['Ticker', 'Date']).sum())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

df.to_csv(
    '../SP500_Historical_Data_cleaned.csv',
    index=False
)

print("\nCleaned dataset saved successfully.")