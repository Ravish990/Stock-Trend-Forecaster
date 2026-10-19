from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import numpy as np
from fastapi.testclient import TestClient

from Backend.main import app
from Backend.services.prediction_service import PredictionService, prediction_service
from Backend.services.company_mapping import CompanyMappingService


class TestStockTrendForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Warmup prediction cache once for all tests
        prediction_service.warmup_all_predictions()
        cls.client = TestClient(app)

    def test_ticker_mapping_coverage(self):
        """Verify all 471 tickers are mapped without duplicates or missing names."""
        mapping_service = CompanyMappingService()
        tickers = mapping_service.load_dataset_tickers()
        self.assertEqual(len(tickers), 471, f"Expected 471 tickers, found {len(tickers)}")
        self.assertEqual(len(set(tickers)), 471, "Found duplicate tickers in universe")

        mapping = mapping_service.build_mapping()
        self.assertEqual(len(mapping), 471)

        # Check key companies
        self.assertIn("AAPL", mapping)
        self.assertEqual(mapping["AAPL"]["company_name"], "Apple Inc.")
        self.assertIn("MSFT", mapping)
        self.assertEqual(mapping["MSFT"]["company_name"], "Microsoft Corporation")

        report = mapping_service.validation_report(valid_tickers_count=471)
        self.assertEqual(report["total_unique_supported_tickers"], 471)
        self.assertEqual(report["successfully_mapped_company_names"], 471)
        self.assertEqual(report["missing_mappings"], 0)
        self.assertEqual(report["duplicate_ticker_checks"], 0)
        self.assertEqual(report["tickers_with_valid_prediction_inputs"], 471)

    def test_probability_boundaries(self):
        """Verify exact confidence boundary classifications from specification."""
        classify = PredictionService._classify_confidence

        # > 60%: UP — High confidence
        self.assertEqual(classify(60.01), "UP — High confidence")
        self.assertEqual(classify(75.5), "UP — High confidence")

        # > 50% and <= 60%: UP — Low confidence
        self.assertEqual(classify(60.0), "UP — Low confidence")
        self.assertEqual(classify(55.0), "UP — Low confidence")
        self.assertEqual(classify(50.01), "UP — Low confidence")

        # == 50%: DOWN — Low confidence
        self.assertEqual(classify(50.0), "DOWN — Low confidence")

        # >= 40% and < 50%: DOWN — Low confidence
        self.assertEqual(classify(49.99), "DOWN — Low confidence")
        self.assertEqual(classify(45.0), "DOWN — Low confidence")
        self.assertEqual(classify(40.0), "DOWN — Low confidence")

        # < 40%: DOWN — High confidence
        self.assertEqual(classify(39.99), "DOWN — High confidence")
        self.assertEqual(classify(25.0), "DOWN — High confidence")
        self.assertEqual(classify(0.0), "DOWN — High confidence")

    def test_target_conversion_and_math(self):
        """Verify log return to price and percentage return calculations."""
        latest_close = 200.0
        log_ret = 0.05
        pred_price = latest_close * np.exp(log_ret)
        pred_pct = ((pred_price / latest_close) - 1.0) * 100.0

        self.assertAlmostEqual(pred_price, 210.2542, places=3)
        self.assertAlmostEqual(pred_pct, 5.1271, places=3)

        # Direction checks
        prob_up_high = 65.0
        dir_high = "UP" if prob_up_high > 50.0 else "DOWN"
        self.assertEqual(dir_high, "UP")

        prob_up_low = 45.0
        dir_low = "UP" if prob_up_low > 50.0 else "DOWN"
        self.assertEqual(dir_low, "DOWN")

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["dataset_available"])
        self.assertEqual(data["total_supported_tickers"], 471)
        self.assertEqual(data["cached_predictions_count"], 471)

    def test_stocks_endpoint(self):
        # 1. Default listing
        response = self.client.get("/api/stocks?page=1&page_size=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 471)
        self.assertEqual(len(data["items"]), 10)
        self.assertEqual(data["page"], 1)

        # 2. Search by company name
        search_res = self.client.get("/api/stocks?search=Apple")
        self.assertEqual(search_res.status_code, 200)
        s_data = search_res.json()
        self.assertGreaterEqual(s_data["total"], 1)
        tickers = [item["ticker"] for item in s_data["items"]]
        self.assertIn("AAPL", tickers)

        # 3. Sort by latest close
        sort_res = self.client.get("/api/stocks?sort_by=latest_close&sort_order=desc&page_size=5")
        self.assertEqual(sort_res.status_code, 200)
        sort_items = sort_res.json()["items"]
        closes = [item["latest_close"] for item in sort_items if item["latest_close"]]
        self.assertEqual(closes, sorted(closes, reverse=True))

    def test_stock_details_and_history(self):
        # Details
        res = self.client.get("/api/stocks/AAPL")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertEqual(data["company_name"], "Apple Inc.")
        self.assertIsNotNone(data["prediction"])
        self.assertGreater(len(data["history_summary"]), 0)

        # History
        hist_res = self.client.get("/api/stocks/AAPL/history?limit=30")
        self.assertEqual(hist_res.status_code, 200)
        hist_data = hist_res.json()
        self.assertEqual(hist_data["ticker"], "AAPL")
        self.assertLessEqual(len(hist_data["history"]), 30)
        self.assertIsNotNone(hist_data["forecast_point"])
        self.assertTrue(hist_data["forecast_point"]["is_forecast"])

    def test_predictions_endpoint(self):
        # Single prediction
        res = self.client.get("/api/predictions/MSFT")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["ticker"], "MSFT")
        self.assertIn(data["predicted_direction"], ["UP", "DOWN"])
        self.assertAlmostEqual(data["probability_up"] + data["probability_down"], 100.0, places=1)

        # Market predictions with filter
        up_res = self.client.get("/api/predictions?direction=UP&min_probability=55&page_size=10")
        self.assertEqual(up_res.status_code, 200)
        up_data = up_res.json()
        for item in up_data["items"]:
            self.assertEqual(item["predicted_direction"], "UP")
            self.assertGreaterEqual(item["probability_up"], 55.0)

    def test_market_summary(self):
        res = self.client.get("/api/market/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_supported_tickers"], 471)
        self.assertEqual(data["successfully_predicted_tickers"], 471)
        self.assertEqual(data["missing_data_tickers"], 0)
        self.assertEqual(data["predicted_up_count"] + data["predicted_down_count"], 471)
        self.assertEqual(data["latest_data_date"], "2026-02-19")

    def test_market_leaderboard(self):
        res = self.client.get("/api/market/leaderboard?category=highest_probability_up&limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["items"]), 10)
        probs = [item["probability_up"] for item in data["items"]]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_model_performance(self):
        res = self.client.get("/api/market/model-performance")
        data = res.json()
        self.assertIn("best_lstm_stock_model", data["model_name"])
        self.assertGreater(data["sample_size"], 0)
        self.assertIn("directional_accuracy_pct", data)
        self.assertIn("calibration", data)
        self.assertGreater(len(data["calibration"]), 0)

    def test_validation_report(self):
        res = self.client.get("/api/market/validation-report")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_unique_supported_tickers"], 471)
        self.assertEqual(data["successfully_mapped_company_names"], 471)
        self.assertEqual(data["missing_mappings"], 0)
        self.assertEqual(data["duplicate_ticker_checks"], 0)
        self.assertEqual(data["tickers_with_valid_prediction_inputs"], 471)
        self.assertEqual(data["missing_data_tickers_count"], 0)
        self.assertEqual(len(data["missing_tickers"]), 0)

    def test_missing_ticker_error_handling(self):
        # 1. Removed ticker SNDK is not found -> 404
        sndk_res = self.client.get("/api/predictions/SNDK")
        self.assertEqual(sndk_res.status_code, 404)
        self.assertIn("not found", sndk_res.json()["detail"].lower())

        # 2. Unknown ticker -> 404
        bad_res = self.client.get("/api/predictions/UNKNOWNXYZ")
        self.assertEqual(bad_res.status_code, 404)


if __name__ == "__main__":
    unittest.main()

