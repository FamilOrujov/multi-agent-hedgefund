
import os
from datetime import datetime
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_LLM_MODEL": "llama3.2",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_DB": "alpha_council_test",
        "TAVILY_API_KEY": "",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


@pytest.fixture
def sample_price_data() -> dict[str, Any]:
    """Sample price data for testing."""
    return {
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "open": [150.0, 152.0, 151.0],
        "high": [153.0, 155.0, 154.0],
        "low": [149.0, 151.0, 150.0],
        "close": [152.0, 154.0, 153.0],
        "volume": [1000000, 1200000, 1100000],
        "latest_close": 153.0,
        "price_change_pct": 0.67,
    }


@pytest.fixture
def sample_company_info() -> dict[str, Any]:
    """Sample company info for testing."""
    return {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "United States",
        "employees": 164000,
        "website": "https://www.apple.com",
        "description": "Apple Inc. designs, manufactures, and markets smartphones...",
    }


@pytest.fixture
def sample_agent_state() -> dict[str, Any]:
    """Sample agent state for testing."""
    return {
        "ticker": "AAPL",
        "analysis_depth": "standard",
        "price_data": {},
        "price_summary": {},
        "company_info": {},
        "financials": {},
        "news_data": [],
        "data_scout_summary": "",
        "technical_indicators": {},
        "technical_signal": {"signal": "Bullish", "confidence": 0.75},
        "technical_analysis": "",
        "financial_ratios": {},
        "health_assessment": {},
        "fundamental_signal": {"signal": "Bullish", "confidence": 0.70},
        "fundamental_analysis": "",
        "sentiment_score": 0.0,
        "sentiment_aggregation": {},
        "sentiment_signal": {"signal": "Neutral", "confidence": 0.55},
        "sentiment_analysis": "",
        "signal_aggregation": {},
        "manager_decision": {
            "decision": "BUY",
            "confidence": 72,
            "position_size": "Moderate",
            "has_consensus": True,
        },
        "final_thesis": "",
        "iteration_count": 1,
        "max_iterations": 5,
        "current_agent": "",
        "status": "completed",
        "human_feedback": None,
        "requires_human_review": False,
        "errors": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


@pytest.fixture
def mock_ollama_client():
    """Mock OllamaClient for testing without Ollama server."""
    with patch("src.llm.ollama_client.ChatOllama") as mock_chat:
        with patch("src.llm.ollama_client.OllamaEmbeddings") as mock_embed:
            mock_chat_instance = MagicMock()
            mock_chat_instance.invoke.return_value = MagicMock(content="Test response")
            mock_chat.return_value = mock_chat_instance

            mock_embed_instance = MagicMock()
            mock_embed_instance.embed_documents.return_value = [[0.1] * 768]
            mock_embed.return_value = mock_embed_instance

            yield {"chat": mock_chat, "embeddings": mock_embed}


@pytest.fixture
def mock_yfinance():
    """Mock yfinance for testing without API calls."""
    import pandas as pd

    with patch("yfinance.Ticker") as mock_ticker:
        mock_stock = MagicMock()

        # Mock history
        mock_stock.history.return_value = pd.DataFrame(
            {
                "Open": [150.0, 152.0, 151.0],
                "High": [153.0, 155.0, 154.0],
                "Low": [149.0, 151.0, 150.0],
                "Close": [152.0, 154.0, 153.0],
                "Volume": [1000000, 1200000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        # Mock info
        mock_stock.info = {
            "shortName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "United States",
            "fullTimeEmployees": 164000,
            "website": "https://www.apple.com",
            "longBusinessSummary": "Apple Inc. designs...",
            "trailingPE": 28.5,
            "forwardPE": 26.0,
            "pegRatio": 2.1,
            "dividendYield": 0.005,
            "beta": 1.2,
            "marketCap": 3000000000000,
            "fiftyTwoWeekLow": 140.0,
            "fiftyTwoWeekHigh": 200.0,
        }

        # Mock financials
        mock_stock.income_stmt = pd.DataFrame()
        mock_stock.balance_sheet = pd.DataFrame()
        mock_stock.cashflow = pd.DataFrame()
        mock_stock.dividends = pd.Series([0.24, 0.24, 0.24])
        mock_stock.recommendations = pd.DataFrame(
            {"To Grade": ["Buy", "Hold", "Buy"]},
            index=pd.date_range("2024-01-01", periods=3),
        )

        mock_ticker.return_value = mock_stock
        yield mock_ticker
