
from typing import Any, Optional

import yfinance as yf
import pandas as pd

from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.data.tavily_client import TavilyClient
from src.data.financial_apis import FinancialDataAggregator


class DataScoutAgent(BaseAgent):
    """Agent responsible for fetching and organizing raw financial data."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.3,
    ):
        super().__init__(ollama_client, model, temperature)
        self.tavily = TavilyClient()
        self.financial_api = FinancialDataAggregator()

    @property
    def name(self) -> str:
        return "data_scout"

    @property
    def role(self) -> str:
        return "Data Scout - Financial Data Retrieval Specialist"

    @property
    def system_prompt(self) -> str:
        return """You are the Data Scout agent in a hedge fund analysis team.

Your responsibilities:
1. Fetch and organize raw financial data from various sources
2. Retrieve historical price data, volume, and basic metrics
3. Gather recent news and announcements related to the ticker
4. Compile company information and key statistics

You provide factual, well-organized data summaries without making investment recommendations.
Focus on accuracy and completeness of data retrieval.

When presenting data:
- Use clear formatting with sections
- Include data timestamps and sources
- Flag any missing or unavailable data
- Highlight unusual patterns in raw data (e.g., volume spikes)
"""

    def fetch_price_data(
        self,
        ticker: str,
        period: str = "3mo",
    ) -> Optional[pd.DataFrame]:
        """Fetch historical price data for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist if not hist.empty else None
        except Exception:
            return None

    def fetch_company_info(self, ticker: str) -> dict[str, Any]:
        """Fetch company information and key statistics."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", None),
                "forward_pe": info.get("forwardPE", None),
                "dividend_yield": info.get("dividendYield", None),
                "beta": info.get("beta", None),
                "52_week_high": info.get("fiftyTwoWeekHigh", None),
                "52_week_low": info.get("fiftyTwoWeekLow", None),
                "avg_volume": info.get("averageVolume", None),
                "description": info.get("longBusinessSummary", "N/A"),
            }
        except Exception:
            return {}

    def fetch_financials(self, ticker: str) -> dict[str, Any]:
        """Fetch financial statements data."""
        try:
            stock = yf.Ticker(ticker)
            return {
                "income_statement": stock.income_stmt.to_dict() if stock.income_stmt is not None else {},
                "balance_sheet": stock.balance_sheet.to_dict() if stock.balance_sheet is not None else {},
                "cash_flow": stock.cashflow.to_dict() if stock.cashflow is not None else {},
            }
        except Exception:
            return {}

    def fetch_news(self, ticker: str) -> list[dict[str, Any]]:
        """Fetch comprehensive news and market research."""
        if not self.tavily.is_configured:
            return []
        
        # Deep search with multiple queries for comprehensive coverage
        all_news = []
        
        # Recent stock news
        news_results = self.tavily.search_news(ticker, max_results=8)
        all_news.extend(news_results)
        
        # Earnings and financial updates
        earnings_results = self.tavily.search_news(ticker, query="earnings report quarterly results", max_results=5)
        all_news.extend(earnings_results)
        
        # Industry and sector news
        sector_results = self.tavily.search_news(ticker, query="industry sector trends", max_results=5)
        all_news.extend(sector_results)
        
        return all_news

    def fetch_market_research(self, ticker: str) -> dict[str, Any]:
        """Fetch deep market research for comprehensive analysis. This data is shared with all analysts."""
        from datetime import datetime
        
        if not self.tavily.is_configured:
            return {"research_available": False, "fetch_timestamp": None}
        
        fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        research = {
            "research_available": True,
            "fetch_timestamp": fetch_time,
            "data_sources_fetched": [],
            "technical_insights": "",
            "fundamental_research": "",
            "sentiment_sources": "",
            "risk_factors": "",
            "competitor_analysis": "",
        }
        
        try:
            # Technical analysis insights from the web
            tech_results = self.tavily.search_analysis(ticker, "technical analysis chart patterns support resistance", max_results=5)
            if tech_results and "error" not in tech_results[0]:
                research["technical_insights"] = "\n".join([r.get("content", "")[:500] for r in tech_results[:3]])
                research["data_sources_fetched"].append(f"technical_insights ({len(tech_results)} results)")
        except Exception:
            pass
        
        try:
            # Fundamental analysis research
            fund_results = self.tavily.search_analysis(ticker, "earnings valuation financial health", max_results=5)
            if fund_results and "error" not in fund_results[0]:
                research["fundamental_research"] = "\n".join([r.get("content", "")[:500] for r in fund_results[:3]])
                research["data_sources_fetched"].append(f"fundamental_research ({len(fund_results)} results)")
        except Exception:
            pass
        
        try:
            # Sentiment and social sources
            sent_results = self.tavily.search_sentiment_sources(ticker, max_results=5)
            if sent_results and "error" not in sent_results[0]:
                research["sentiment_sources"] = "\n".join([r.get("content", "")[:400] for r in sent_results[:3]])
                research["data_sources_fetched"].append(f"sentiment_sources ({len(sent_results)} results)")
        except Exception:
            pass
        
        try:
            # Risk factors
            risk_results = self.tavily.search_analysis(ticker, "risks concerns challenges headwinds", max_results=5)
            if risk_results and "error" not in risk_results[0]:
                research["risk_factors"] = "\n".join([r.get("content", "")[:400] for r in risk_results[:3]])
                research["data_sources_fetched"].append(f"risk_factors ({len(risk_results)} results)")
        except Exception:
            pass
        
        try:
            # Competitor analysis
            comp_results = self.tavily.search_analysis(ticker, "competitors industry comparison market share", max_results=5)
            if comp_results and "error" not in comp_results[0]:
                research["competitor_analysis"] = "\n".join([r.get("content", "")[:400] for r in comp_results[:3]])
                research["data_sources_fetched"].append(f"competitor_analysis ({len(comp_results)} results)")
        except Exception:
            pass
        
        return research

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        """Fetch all relevant data for the given ticker."""
        ticker = state.get("ticker", "")
        if not ticker:
            return {**state, "data_scout_error": "No ticker provided"}

        stock_data = self.financial_api.get_stock_data(ticker)
        
        price_data = self.fetch_price_data(ticker)
        company_info = stock_data.get("company_info", self.fetch_company_info(ticker))
        financials = stock_data.get("financials", self.fetch_financials(ticker))
        valuation = stock_data.get("valuation", {})
        recommendations = stock_data.get("recommendations", {})
        
        # Fetch comprehensive news and deep market research
        news_data = self.fetch_news(ticker)
        market_research = self.fetch_market_research(ticker)

        price_summary = {}
        if price_data is not None and not price_data.empty:
            price_summary = {
                "latest_close": float(price_data["Close"].iloc[-1]),
                "period_high": float(price_data["High"].max()),
                "period_low": float(price_data["Low"].min()),
                "avg_volume": float(price_data["Volume"].mean()),
                "price_change_pct": float(
                    (price_data["Close"].iloc[-1] - price_data["Close"].iloc[0])
                    / price_data["Close"].iloc[0]
                    * 100
                ),
                "data_points": len(price_data),
            }

        news_summary = ""
        if news_data:
            news_titles = [n.get("title", "") for n in news_data if "title" in n][:8]
            news_summary = "; ".join(news_titles)

        summary_prompt = f"""Summarize the following data for {ticker}:

Company Info: {company_info}
Price Summary: {price_summary}
Valuation Metrics: {valuation}
Analyst Recommendations: {recommendations}
Recent News Headlines: {news_summary}

Provide a concise data summary highlighting key metrics and any notable observations."""

        data_summary = self.invoke(summary_prompt)

        # Convert DataFrame to dict format that can be reconstructed
        price_data_dict = {}
        if price_data is not None and not price_data.empty:
            # Use 'list' orient for proper reconstruction
            price_data_dict = {
                "Close": price_data["Close"].tolist(),
                "Open": price_data["Open"].tolist(),
                "High": price_data["High"].tolist(),
                "Low": price_data["Low"].tolist(),
                "Volume": price_data["Volume"].tolist(),
                "index": price_data.index.tolist(),
            }
        
        return {
            **state,
            "price_data": price_data_dict,
            "price_summary": price_summary,
            "company_info": company_info,
            "financials": financials,
            "valuation": valuation,
            "recommendations": recommendations,
            "news_data": [n.get("title", "") for n in news_data] if news_data else [],
            "data_scout_summary": data_summary,
            # Deep market research passed to all agents
            "market_research": market_research,
        }
