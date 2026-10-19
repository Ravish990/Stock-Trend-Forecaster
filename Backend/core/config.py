from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "merged_news_target_data.csv"

# Automatically use best available model checkpoint
V4_MODEL = ROOT_DIR / "best_lstm_stock_model_v4_h5.pth"
V4_SCALER = ROOT_DIR / "lstm_stock_scalers_V4_h5.pkl"
V3_MODEL = ROOT_DIR / "best_lstm_stock_model_v3_h5.pth"
V3_SCALER = ROOT_DIR / "lstm_stock_scalers_V3_h5.pkl"

MODEL_PATH = V4_MODEL if V4_MODEL.exists() else V3_MODEL
SCALER_PATH = V4_SCALER if V4_SCALER.exists() else V3_SCALER

COMPANY_METADATA_PATH = ROOT_DIR / "data" / "company_metadata.csv"
MODEL_PERFORMANCE_PATH = ROOT_DIR / "data" / "model_performance.json"
AAPL_COMPARISON_PATH = ROOT_DIR / "aapl_feb2026_forecast_comparison.csv"
