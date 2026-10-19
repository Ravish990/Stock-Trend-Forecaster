from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "merged_news_target_data.csv"
MODEL_PATH = ROOT_DIR / "best_lstm_stock_model_v3_h5.pth"
SCALER_PATH = ROOT_DIR / "lstm_stock_scalers_V3_h5.pkl"
COMPANY_METADATA_PATH = ROOT_DIR / "data" / "company_metadata.csv"
MODEL_PERFORMANCE_PATH = ROOT_DIR / "data" / "model_performance.json"
AAPL_COMPARISON_PATH = ROOT_DIR / "aapl_feb2026_forecast_comparison.csv"
