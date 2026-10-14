import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from tqdm import tqdm

from Model.model import LSTMModel
from Model.train_model import (
    FEATURES,
    MODEL_PATH,
    SCALER_PATH,
    SEQUENCE,
    device,
)


def run_screener():
    print("Loading cross-sectional market data...")
    data = pd.read_csv("merged_news_target_data.csv")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    
    latest_date = data["Date"].max()
    print(f"Screening market based on latest available data: {latest_date.date()}")
    
    print("Loading global LSTM model and scaler...")
    scaler = joblib.load(SCALER_PATH)
    model = LSTMModel(
        input_size=len(FEATURES),
        hidden_size=32,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    trade_signals = []
    grouped = data.groupby("Ticker")
    
    print(f"Scanning {len(grouped)} tickers for high-conviction setups...")
    
    with torch.no_grad():
        for ticker, df in tqdm(grouped, desc="Processing Tickers", unit="stock"):
            df = df.sort_values("Date").reset_index(drop=True)
            
            if len(df) < SEQUENCE:
                continue
                
            recent_df = df.tail(SEQUENCE).copy()
            
            try:
                scaled_features = scaler.transform(recent_df[FEATURES])
            except Exception:
                continue 
                
            X_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)
            
            pred_ret_log, pred_dir_logit = model(X_tensor)
            prob_up = torch.sigmoid(pred_dir_logit).item()
            pred_ret = pred_ret_log.item()
            
            current_price = df["Close"].iloc[-1]
            target_price = current_price * np.exp(pred_ret)
            
            trade_signals.append({
                "Ticker": ticker,
                "Current_Price": current_price,
                "Target_Price": target_price,
                "Prob_Up": prob_up
            })

    results_df = pd.DataFrame(trade_signals)
    
    # Filter for the proven statistical edge (>60% Confidence)
    longs = results_df[results_df["Prob_Up"] > 0.60].sort_values("Prob_Up", ascending=False)
    
    print("\n" + "="*75)
    print("CONSTRUCTING SECTOR-NEUTRAL PORTFOLIO")
    print("="*75)
    
    # --- RISK MANAGEMENT LIMITS ---
    MAX_POSITIONS = 10
    MAX_PER_SECTOR = 2
    
    final_portfolio = []
    sector_counts = {}
    
    print("Fetching live sector data for top candidates from Yahoo Finance...")
    
    for _, row in longs.iterrows():
        if len(final_portfolio) >= MAX_POSITIONS:
            break
            
        ticker = row["Ticker"]
        
        try:
            # Fetch sector dynamically using yfinance info dictionary
            ticker_obj = yf.Ticker(ticker)
            sector = ticker_obj.info.get('sector', 'Unknown')
        except Exception:
            sector = 'Unknown'
            
        if sector not in sector_counts:
            sector_counts[sector] = 0
            
        # Only add the stock if we haven't hit the sector limit
        if sector_counts[sector] < MAX_PER_SECTOR or sector == 'Unknown':
            final_portfolio.append({
                "Ticker": ticker,
                "Sector": sector,
                "Conf": row["Prob_Up"],
                "Price": row["Current_Price"],
                "Target": row["Target_Price"]
            })
            if sector != 'Unknown':
                sector_counts[sector] += 1

    print(f"\n🟢 TOP DIVERSIFIED LONGS (Target Size: {MAX_POSITIONS}, Max {MAX_PER_SECTOR} per sector)")
    print("-" * 75)
    if final_portfolio:
        for p in final_portfolio:
            print(f"{p['Ticker']:<6} | Sector: {p['Sector']:<23} | Conf: {p['Conf']*100:>5.2f}% | Price: ${p['Price']:>7.2f}")
    else:
        print("No long setups met the strict 60% threshold today.")
    print("="*75)

if __name__ == "__main__":
    run_screener()