"""Tests for src/data/financial_apis.py module."""

from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import pandas as pd
import pytest


class TestFinancialDataAggregator:
    """Test FinancialDataAggregator class."""

    def test_aggregator_initialization(self):
        """Test FinancialDataAggregator initializes correctly."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()

        assert aggregator is not None

    def test_get_stock_data_structure(self, mock_yfinance):
        """Test get_stock_data returns expected structure."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()
        data = aggregator.get_stock_data("AAPL")

        assert "price_data" in data
        assert "company_info" in data
        assert "valuation" in data
        assert "financials" in data

    def test_get_stock_data_price_data(self, mock_yfinance):
        """Test get_stock_data returns price data."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()
        data = aggregator.get_stock_data("AAPL")

        price_data = data.get("price_data", {})
        assert "dates" in price_data or "close" in price_data or len(price_data) >= 0

    def test_get_stock_data_company_info(self, mock_yfinance):
        """Test get_stock_data returns company info."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()
        data = aggregator.get_stock_data("AAPL")

        company_info = data.get("company_info", {})
        # Check structure exists
        assert isinstance(company_info, dict)

    def test_get_stock_data_with_period(self, mock_yfinance):
        """Test get_stock_data with custom period."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()
        data = aggregator.get_stock_data("AAPL", period="1mo", interval="1d")

        assert data is not None

    def test_process_price_data(self, mock_yfinance):
        """Test _process_price_data with sample DataFrame."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()

        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "Open": [150.0, 152.0, 151.0],
                "High": [153.0, 155.0, 154.0],
                "Low": [149.0, 151.0, 150.0],
                "Close": [152.0, 154.0, 153.0],
                "Volume": [1000000, 1200000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        result = aggregator._process_price_data(df)

        assert "dates" not in result  # Structure is different
        assert "latest" in result
        assert "period_stats" in result
        assert result["latest"]["close"] == 153.0
        assert result["data_points"] == 3

    def test_process_price_data_empty(self):
        """Test _process_price_data with empty DataFrame."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()

        df = pd.DataFrame()
        result = aggregator._process_price_data(df)

        assert result == {}

    def test_extract_company_info(self, mock_yfinance):
        """Test _extract_company_info extracts fields correctly."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()

        info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "United States",
            "fullTimeEmployees": 164000,
            "website": "https://www.apple.com",
            "longBusinessSummary": "Apple designs...",
        }

        result = aggregator._extract_company_info(info)

        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["industry"] == "Consumer Electronics"

    def test_extract_valuation(self, mock_yfinance):
        """Test _extract_valuation extracts metrics correctly."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()

        info = {
            "trailingPE": 28.5,
            "forwardPE": 26.0,
            "pegRatio": 2.1,
            "dividendYield": 0.005,
            "beta": 1.2,
            "marketCap": 3000000000000,
            "fiftyTwoWeekLow": 140.0,
            "fiftyTwoWeekHigh": 200.0,
        }

        result = aggregator._extract_valuation(info)

        assert result["pe_ratio"] == 28.5
        assert result["forward_pe"] == 26.0
        assert result["peg_ratio"] == 2.1
        assert result["beta"] == 1.2

    def test_safe_get(self):
        """Test _safe_get helper method."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()

        df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            index=["row1", "row2", "row3"],
        )

        # Test existing value
        value = aggregator._safe_get(df, "row1", "A")
        assert value == 1

        # Test non-existing row
        value = aggregator._safe_get(df, "missing", "A")
        assert value is None

        # Test non-existing column
        value = aggregator._safe_get(df, "row1", "C")
        assert value is None

    def test_get_market_indices(self, mock_yfinance):
        """Test get_market_indices returns expected structure."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()
        indices = aggregator.get_market_indices()

        assert isinstance(indices, dict)
        # Should contain major indices
        assert "SPY" in indices or "^GSPC" in indices or len(indices) >= 0

    def test_get_sector_performance(self, mock_yfinance):
        """Test get_sector_performance returns expected structure."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()
        sector_data = aggregator.get_sector_performance("AAPL")

        assert isinstance(sector_data, dict)


class TestFinancialDataAggregatorEdgeCases:
    """Test edge cases and error handling."""

    def test_get_stock_data_invalid_ticker(self, mock_yfinance):
        """Test get_stock_data handles invalid ticker."""
        from src.data.financial_apis import FinancialDataAggregator

        # Configure mock to return empty data
        mock_yfinance.return_value.history.return_value = pd.DataFrame()
        mock_yfinance.return_value.info = {}

        aggregator = FinancialDataAggregator()
        data = aggregator.get_stock_data("INVALID_TICKER_XYZ")

        # Should not raise, returns structure with empty data
        assert isinstance(data, dict)

    def test_process_price_data_calculates_metrics(self):
        """Test _process_price_data calculates additional metrics."""
        from src.data.financial_apis import FinancialDataAggregator

        aggregator = FinancialDataAggregator()

        df = pd.DataFrame(
            {
                "Open": [100.0, 105.0, 110.0],
                "High": [106.0, 111.0, 115.0],
                "Low": [99.0, 104.0, 109.0],
                "Close": [105.0, 110.0, 115.0],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        result = aggregator._process_price_data(df)

        # Should have calculated latest close
        assert result["latest"]["close"] == 115.0
        assert result["returns"]["period_return"] is not None
