export interface PredictionSnapshot {
  ticker: string;
  company_name?: string | null;
  latest_close?: number | null;
  latest_data_date?: string | null;
  predicted_price_5d?: number | null;
  predicted_return_pct?: number | null;
  probability_up?: number | null;
  probability_down?: number | null;
  predicted_direction?: 'UP' | 'DOWN' | null;
  confidence_label?: string | null;
  prediction_date?: string | null;
  forecast_horizon: number;
  model_name?: string | null;
  sector?: string | null;
  industry?: string | null;
}

export interface MarketPredictionsResponse {
  items: PredictionSnapshot[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  generated_at?: string | null;
}

export interface MarketSummaryResponse {
  total_supported_tickers: number;
  successfully_predicted_tickers: number;
  missing_data_tickers: number;
  predicted_up_count: number;
  predicted_down_count: number;
  avg_predicted_return: number;
  highest_predicted_probability: number;
  lowest_predicted_probability: number;
  highest_probability_ticker?: string | null;
  latest_data_date: string;
  prediction_generated_at: string;
  model_name: string;
  forecast_horizon: number;
}

export interface LeaderboardResponse {
  category: string;
  title: string;
  description: string;
  items: PredictionSnapshot[];
  total: number;
  generated_at?: string | null;
}

export interface CalibrationBin {
  bin_range: string;
  predicted_prob: number;
  actual_freq: number;
  count: number;
}

export interface ModelPerformanceResponse {
  evaluation_period: string;
  sample_size: number;
  evaluated_tickers_count: number;
  model_name: string;
  forecast_horizon: string;
  directional_accuracy_pct: number;
  precision: number;
  recall: number;
  brier_score: number;
  log_loss: number;
  rmse_log_return: number;
  rmse_return_pct: number;
  up_fraction_pct: number;
  majority_baseline_pct: number;
  calibration: CalibrationBin[];
  description: string;
}

