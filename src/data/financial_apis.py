
from typing import Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

from src.config.settings import get_settings


class FinancialDataAggregator:
    """
    Aggregates data from multiple free financial APIs.
    
    Free sources used:
    - yfinance: Price data, company info, financials (Yahoo Finance)
    - FRED API: Economic indicators (optional, requires free API key)
    - SEC EDGAR: Company filings (free, no API key needed)
    
    Note: yfinance scrapes Yahoo Finance. For production, consider
    paid APIs like Tiingo, Alpha Vantage, or Polygon.io
    """

    def __init__(self):
        self.settings = get_settings()

    def get_stock_data(
        self,
        ticker: str,
        period: str = "3mo",
        interval: str = "1d",
    ) -> dict[str, Any]:
        """
        Get comprehensive stock data from yfinance.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary with price data, company info, and financials
        """
        try:
            stock = yf.Ticker(ticker)
            
            hist = stock.history(period=period, interval=interval)
            
            info = stock.info
            
            result = {
                "ticker": ticker,
                "price_data": self._process_price_data(hist),
                "company_info": self._extract_company_info(info),
                "valuation": self._extract_valuation(info),
                "financials": self._get_financials(stock),
                "dividends": self._get_dividends(stock),
                "recommendations": self._get_recommendations(stock),
            }
            
            return result
            
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    def _process_price_data(self, hist: pd.DataFrame) -> dict[str, Any]:
        """Process historical price data."""
        if hist.empty:
            return {}
            
        return {
            "latest": {
                "open": float(hist["Open"].iloc[-1]),
                "high": float(hist["High"].iloc[-1]),
                "low": float(hist["Low"].iloc[-1]),
                "close": float(hist["Close"].iloc[-1]),
                "volume": int(hist["Volume"].iloc[-1]),
            },
            "period_stats": {
                "high": float(hist["High"].max()),
                "low": float(hist["Low"].min()),
                "avg_volume": float(hist["Volume"].mean()),
                "volatility": float(hist["Close"].pct_change().std() * 100),
            },
            "returns": {
                "period_return": float(
                    (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) 
                    / hist["Close"].iloc[0] * 100
                ),
            },
            "data_points": len(hist),
            "raw_data": hist.to_dict(),
        }

    def _extract_company_info(self, info: dict) -> dict[str, Any]:
        """Extract company information."""
        return {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "website": info.get("website", "N/A"),
            "employees": info.get("fullTimeEmployees", 0),
            "description": info.get("longBusinessSummary", ""),
            "exchange": info.get("exchange", "N/A"),
            "currency": info.get("currency", "USD"),
        }

    def _extract_valuation(self, info: dict) -> dict[str, Any]:
        """Extract valuation metrics."""
        return {
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": info.get("enterpriseValue", 0),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            "beta": info.get("beta"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "50_day_avg": info.get("fiftyDayAverage"),
            "200_day_avg": info.get("twoHundredDayAverage"),
        }

    def _get_financials(self, stock: yf.Ticker) -> dict[str, Any]:
        """Get financial statements."""
        try:
            financials = {}
            
            if stock.income_stmt is not None and not stock.income_stmt.empty:
                income = stock.income_stmt
                latest_col = income.columns[0]
                financials["income_statement"] = {
                    "total_revenue": self._safe_get(income, "Total Revenue", latest_col),
                    "gross_profit": self._safe_get(income, "Gross Profit", latest_col),
                    "operating_income": self._safe_get(income, "Operating Income", latest_col),
                    "net_income": self._safe_get(income, "Net Income", latest_col),
                    "ebitda": self._safe_get(income, "EBITDA", latest_col),
                }
            
            if stock.balance_sheet is not None and not stock.balance_sheet.empty:
                balance = stock.balance_sheet
                latest_col = balance.columns[0]
                financials["balance_sheet"] = {
                    "total_assets": self._safe_get(balance, "Total Assets", latest_col),
                    "total_liabilities": self._safe_get(balance, "Total Liabilities Net Minority Interest", latest_col),
                    "total_equity": self._safe_get(balance, "Stockholders Equity", latest_col),
                    "cash": self._safe_get(balance, "Cash And Cash Equivalents", latest_col),
                    "total_debt": self._safe_get(balance, "Total Debt", latest_col),
                }
            
            if stock.cashflow is not None and not stock.cashflow.empty:
                cashflow = stock.cashflow
                latest_col = cashflow.columns[0]
                financials["cash_flow"] = {
                    "operating_cash_flow": self._safe_get(cashflow, "Operating Cash Flow", latest_col),
                    "investing_cash_flow": self._safe_get(cashflow, "Investing Cash Flow", latest_col),
                    "financing_cash_flow": self._safe_get(cashflow, "Financing Cash Flow", latest_col),
                    "free_cash_flow": self._safe_get(cashflow, "Free Cash Flow", latest_col),
                }
            
            return financials
            
        except Exception:
            return {}

    def _safe_get(self, df: pd.DataFrame, row: str, col: Any) -> Optional[float]:
        """Safely get a value from a DataFrame."""
        try:
            if row in df.index:
                val = df.loc[row, col]
                return float(val) if pd.notna(val) else None
        except Exception:
            pass
        return None

    def _get_dividends(self, stock: yf.Ticker) -> dict[str, Any]:
        """Get dividend information."""
        try:
            info = stock.info
            return {
                "dividend_rate": info.get("dividendRate"),
                "dividend_yield": info.get("dividendYield"),
                "ex_dividend_date": info.get("exDividendDate"),
                "payout_ratio": info.get("payoutRatio"),
                "five_year_avg_yield": info.get("fiveYearAvgDividendYield"),
            }
        except Exception:
            return {}

    def _get_recommendations(self, stock: yf.Ticker) -> dict[str, Any]:
        """Get analyst recommendations."""
        try:
            info = stock.info
            return {
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "target_mean": info.get("targetMeanPrice"),
                "target_median": info.get("targetMedianPrice"),
                "recommendation": info.get("recommendationKey"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
            }
        except Exception:
            return {}

    def get_market_indices(self) -> dict[str, Any]:
        """Get major market indices data."""
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^VIX": "VIX",
        }
        
        result = {}
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    result[name] = {
                        "price": float(hist["Close"].iloc[-1]),
                        "change": float(
                            (hist["Close"].iloc[-1] - hist["Close"].iloc[-2])
                            / hist["Close"].iloc[-2] * 100
                        ),
                    }
            except Exception:
                continue
                
        return result

    def get_sector_performance(self, ticker: str) -> dict[str, Any]:
        """Get sector performance comparison."""
        try:
            stock = yf.Ticker(ticker)
            sector = stock.info.get("sector", "")
            
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financial Services": "XLF",
                "Consumer Cyclical": "XLY",
                "Consumer Defensive": "XLP",
                "Energy": "XLE",
                "Industrials": "XLI",
                "Basic Materials": "XLB",
                "Real Estate": "XLRE",
                "Utilities": "XLU",
                "Communication Services": "XLC",
            }
            
            if sector in sector_etfs:
                etf = yf.Ticker(sector_etfs[sector])
                hist = etf.history(period="1mo")
                if not hist.empty:
                    return {
                        "sector": sector,
                        "etf": sector_etfs[sector],
                        "performance_1m": float(
                            (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                            / hist["Close"].iloc[0] * 100
                        ),
                    }
            
            return {"sector": sector}
            
        except Exception:
            return {}
