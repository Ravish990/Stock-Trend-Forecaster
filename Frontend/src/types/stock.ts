export interface StockMetadata {
  ticker: string;
  company_name: string | null;
  sector: string | null;
  industry: string | null;
  latest_close: number | null;
  latest_data_date: string | null;
  predicted_price_5d?: number | null;
  predicted_return_pct?: number | null;
  probability_up?: number | null;
  probability_down?: number | null;
  predicted_direction?: 'UP' | 'DOWN' | null;
  confidence_label?: string | null;
  prediction_available: boolean;
}

export interface StockListResponse {
  items: StockMetadata[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  search?: string | null;
  sort_by: string;
  sort_order: string;
}

export interface StockHistoryItem {
  date: string;
  close: number;
  open?: number | null;
  high?: number | null;
  low?: number | null;
  volume?: number | null;
  predicted_price?: number | null;
  hit?: boolean | null;
  is_forecast?: boolean;
}

export interface StockHistoryResponse {
  ticker: string;
  company_name: string | null;
  latest_close: number;
  latest_data_date: string;
  history: StockHistoryItem[];
  forecast_point?: StockHistoryItem | null;
}

export interface StockDetailResponse {
  ticker: string;
  company_name: string | null;
  sector: string | null;
  industry: string | null;
  latest_close: number;
  latest_data_date: string;
  prediction?: import('./prediction').PredictionSnapshot | null;
  history_summary: StockHistoryItem[];
}

export interface MissingTickerDetail {
  ticker: string;
  company_name?: string | null;
  sector?: string | null;
  industry?: string | null;
  reason: string;
}

export interface ValidationReportResponse {
  total_unique_supported_tickers: number;
  successfully_mapped_company_names: number;
  missing_mappings: number;
  duplicate_ticker_checks: number;
  tickers_with_valid_prediction_inputs: number;
  missing_data_tickers_count: number;
  missing_tickers: MissingTickerDetail[];
}
