import axios from 'axios';
import {
  StockDetailResponse,
  StockHistoryResponse,
  StockListResponse,
  ValidationReportResponse,
} from '../types/stock';
import {
  LeaderboardResponse,
  MarketPredictionsResponse,
  MarketSummaryResponse,
  ModelPerformanceResponse,
  PredictionSnapshot,
} from '../types/prediction';

const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || '/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // System Health
  async getHealth() {
    const { data } = await apiClient.get('/health');
    return data;
  },

  // Stock Directory
  async getStocks(params?: {
    search?: string;
    sector?: string;
    direction?: string;
    confidence?: string;
    page?: number;
    page_size?: number;
    sort_by?: string;
    sort_order?: string;
  }): Promise<StockListResponse> {
    const { data } = await apiClient.get<StockListResponse>('/stocks', { params });
    return data;
  },

  // Stock Details
  async getStockDetails(ticker: string): Promise<StockDetailResponse> {
    const { data } = await apiClient.get<StockDetailResponse>(`/stocks/${encodeURIComponent(ticker)}`);
    return data;
  },

  // Stock History & Chart
  async getStockHistory(ticker: string, limit: number = 90): Promise<StockHistoryResponse> {
    const { data } = await apiClient.get<StockHistoryResponse>(`/stocks/${encodeURIComponent(ticker)}/history`, {
      params: { limit },
    });
    return data;
  },

  // Individual Prediction
  async getPrediction(ticker: string): Promise<PredictionSnapshot> {
    const { data } = await apiClient.get<PredictionSnapshot>(`/predictions/${encodeURIComponent(ticker)}`);
    return data;
  },

  // Market Predictions
  async getPredictions(params?: {
    direction?: string;
    min_probability?: number;
    max_probability?: number;
    min_return?: number;
    max_return?: number;
    confidence?: string;
    search?: string;
    sort_by?: string;
    sort_order?: string;
    page?: number;
    page_size?: number;
  }): Promise<MarketPredictionsResponse> {
    const { data } = await apiClient.get<MarketPredictionsResponse>('/predictions', { params });
    return data;
  },

  // Market Summary
  async getMarketSummary(): Promise<MarketSummaryResponse> {
    const { data } = await apiClient.get<MarketSummaryResponse>('/market/summary');
    return data;
  },

  // Leaderboard
  async getLeaderboard(category: string = 'highest_probability_up', limit: number = 20): Promise<LeaderboardResponse> {
    const { data } = await apiClient.get<LeaderboardResponse>('/market/leaderboard', {
      params: { category, limit },
    });
    return data;
  },

  // Model Performance
  async getModelPerformance(): Promise<ModelPerformanceResponse> {
    const { data } = await apiClient.get<ModelPerformanceResponse>('/market/model-performance');
    return data;
  },

  // Validation Report
  async getValidationReport(): Promise<ValidationReportResponse> {
    const { data } = await apiClient.get<ValidationReportResponse>('/market/validation-report');
    return data;
  },
};
