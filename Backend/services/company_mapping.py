from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Backend.core.config import COMPANY_METADATA_PATH, DATASET_PATH


class CompanyMappingService:
    def __init__(self, dataset_path: str | Path = DATASET_PATH, metadata_path: str | Path = COMPANY_METADATA_PATH):
        self.dataset_path = Path(dataset_path)
        self.metadata_path = Path(metadata_path)
        self._metadata_cache: dict[str, dict[str, str | None]] = {}
        self._mapping_loaded = False

    def _ensure_metadata_file(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            return

        default = pd.DataFrame(columns=["ticker", "company_name", "sector", "industry"])
        default.to_csv(self.metadata_path, index=False)

    def load_dataset_tickers(self) -> list[str]:
        if self.metadata_path.exists():
            meta = self._load_cached_metadata()
            if meta:
                return sorted([t for t in meta.keys() if t != "SNDK"])

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path, usecols=["Ticker"])
        tickers = pd.Series(df["Ticker"].astype(str).str.strip().str.upper()).dropna().unique().tolist()
        return sorted([t for t in tickers if t != "SNDK"])

    def _load_cached_metadata(self) -> dict[str, dict[str, str | None]]:
        self._ensure_metadata_file()
        if not self.metadata_path.exists():
            return {}

        df = pd.read_csv(self.metadata_path)
        if df.empty:
            return {}

        result: dict[str, dict[str, str | None]] = {}
        for row in df.to_dict(orient="records"):
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            result[ticker] = {
                "company_name": row.get("company_name"),
                "sector": row.get("sector"),
                "industry": row.get("industry"),
            }
        return result

    def build_mapping(self) -> dict[str, dict[str, str | None]]:
        if self._mapping_loaded:
            return self._metadata_cache

        dataset_tickers = self.load_dataset_tickers()
        metadata = self._load_cached_metadata()
        for ticker in dataset_tickers:
            metadata.setdefault(ticker, {"company_name": None, "sector": None, "industry": None})
        self._metadata_cache = metadata
        self._mapping_loaded = True
        return metadata

    def get_company_details(self, ticker: str) -> dict[str, str | None]:
        key = ticker.strip().upper()
        mapping = self.build_mapping()
        metadata = mapping.get(key, {"company_name": None, "sector": None, "industry": None})
        return {
            "ticker": key,
            "company_name": metadata.get("company_name"),
            "sector": metadata.get("sector"),
            "industry": metadata.get("industry"),
        }

    def validation_report(self, valid_tickers_count: int | None = None) -> dict[str, Any]:
        mapping = self.build_mapping()
        dataset_tickers = self.load_dataset_tickers()
        unique = sorted(set(dataset_tickers))
        mapped = sum(1 for ticker in unique if mapping.get(ticker, {}).get("company_name"))
        missing = len(unique) - mapped
        duplicates = len(dataset_tickers) - len(set(dataset_tickers))
        valid_inputs = valid_tickers_count if valid_tickers_count is not None else len(unique)
        missing_count = max(0, len(unique) - valid_inputs)
        return {
            "total_unique_supported_tickers": len(unique),
            "successfully_mapped_company_names": mapped,
            "missing_mappings": missing,
            "duplicate_ticker_checks": duplicates,
            "tickers_with_valid_prediction_inputs": valid_inputs,
            "missing_data_tickers_count": missing_count,
            "missing_tickers": [],
        }
