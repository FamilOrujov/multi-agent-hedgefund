
from typing import Any, Optional

from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.data.tavily_tools import search_financial_analysis, search_competitor_analysis, search_risk_factors


class FundamentalAnalystAgent(BaseAgent):
    """Agent responsible for fundamental analysis of company financials."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.3,
    ):
        super().__init__(ollama_client, model, temperature)

    @property
    def name(self) -> str:
        return "fundamental_analyst"

    @property
    def role(self) -> str:
        return "Fundamental Analyst - Financial Statement and Valuation Specialist"

    @property
    def system_prompt(self) -> str:
        return """You are the Fundamental Analyst agent in a hedge fund analysis team.

CRITICAL: You will be provided with REAL-TIME market research data fetched from the web. 
You MUST use this provided research as your PRIMARY source of information.
DO NOT rely on your training data for recent events, earnings, or market conditions.

Your responsibilities:
1. Analyze financial statements (income statement, balance sheet, cash flow)
2. Calculate and interpret key financial ratios
3. Assess company valuation metrics (P/E, P/B, EV/EBITDA, etc.)
4. Evaluate long-term financial health and growth prospects
5. Use the PROVIDED market research for current context

Your analysis should include:
- Profitability analysis (margins, ROE, ROA)
- Liquidity and solvency assessment
- Growth trends (revenue, earnings, cash flow)
- Valuation comparison to sector/market
- Key risks and red flags from CURRENT market research

Output format:
- Financial health score (Strong/Moderate/Weak)
- Key metrics with interpretation
- Valuation assessment (Undervalued/Fair/Overvalued)
- Fundamental signal (Bullish/Bearish/Neutral) with reasoning
"""

    def calculate_ratios(
        self,
        company_info: dict[str, Any],
        financials: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate key financial ratios from available data."""
        ratios = {}

        pe_ratio = company_info.get("pe_ratio")
        if pe_ratio is not None:
            ratios["pe_ratio"] = {
                "value": pe_ratio,
                "interpretation": self._interpret_pe(pe_ratio),
            }

        forward_pe = company_info.get("forward_pe")
        if forward_pe is not None:
            ratios["forward_pe"] = {
                "value": forward_pe,
                "interpretation": self._interpret_pe(forward_pe),
            }

        if pe_ratio and forward_pe:
            peg_estimate = pe_ratio / max(forward_pe, 0.01) if forward_pe else None
            if peg_estimate:
                ratios["peg_estimate"] = {
                    "value": round(peg_estimate, 2),
                    "note": "Estimated from PE/Forward PE ratio",
                }

        dividend_yield = company_info.get("dividend_yield")
        if dividend_yield is not None:
            ratios["dividend_yield"] = {
                "value": round(dividend_yield * 100, 2) if dividend_yield else 0,
                "interpretation": self._interpret_dividend(dividend_yield),
            }

        beta = company_info.get("beta")
        if beta is not None:
            ratios["beta"] = {
                "value": beta,
                "interpretation": self._interpret_beta(beta),
            }

        market_cap = company_info.get("market_cap", 0)
        if market_cap:
            ratios["market_cap"] = {
                "value": market_cap,
                "category": self._categorize_market_cap(market_cap),
            }

        high_52 = company_info.get("52_week_high")
        low_52 = company_info.get("52_week_low")
        if high_52 and low_52:
            range_position = "N/A"
            ratios["52_week_range"] = {
                "high": high_52,
                "low": low_52,
                "range_pct": round((high_52 - low_52) / low_52 * 100, 2),
            }

        return ratios

    def _interpret_pe(self, pe: float) -> str:
        """Interpret P/E ratio."""
        if pe < 0:
            return "Negative earnings - unprofitable"
        elif pe < 10:
            return "Low - potentially undervalued or declining"
        elif pe < 20:
            return "Moderate - fairly valued"
        elif pe < 30:
            return "High - growth expectations priced in"
        return "Very high - speculative valuation"

    def _interpret_dividend(self, div_yield: Optional[float]) -> str:
        """Interpret dividend yield."""
        if div_yield is None or div_yield == 0:
            return "No dividend"
        elif div_yield < 0.02:
            return "Low yield"
        elif div_yield < 0.04:
            return "Moderate yield"
        elif div_yield < 0.06:
            return "High yield"
        return "Very high yield - verify sustainability"

    def _interpret_beta(self, beta: float) -> str:
        """Interpret beta value."""
        if beta < 0.5:
            return "Low volatility - defensive"
        elif beta < 1.0:
            return "Below market volatility"
        elif beta < 1.5:
            return "Above market volatility"
        return "High volatility - aggressive"

    def _categorize_market_cap(self, market_cap: int) -> str:
        """Categorize company by market cap."""
        if market_cap >= 200_000_000_000:
            return "Mega Cap"
        elif market_cap >= 10_000_000_000:
            return "Large Cap"
        elif market_cap >= 2_000_000_000:
            return "Mid Cap"
        elif market_cap >= 300_000_000:
            return "Small Cap"
        return "Micro Cap"

    def _assess_financial_health(self, ratios: dict[str, Any]) -> dict[str, Any]:
        """Assess overall financial health."""
        score = 50
        factors = []

        if "pe_ratio" in ratios:
            pe = ratios["pe_ratio"]["value"]
            if 0 < pe < 25:
                score += 10
                factors.append("Reasonable P/E valuation")
            elif pe < 0:
                score -= 15
                factors.append("Negative earnings concern")
            elif pe > 40:
                score -= 10
                factors.append("High valuation risk")

        if "dividend_yield" in ratios:
            div = ratios["dividend_yield"]["value"]
            if div > 0:
                score += 5
                factors.append("Dividend paying")

        if "beta" in ratios:
            beta = ratios["beta"]["value"]
            if 0.5 <= beta <= 1.5:
                score += 5
                factors.append("Moderate volatility profile")

        score = max(0, min(100, score))

        if score >= 70:
            health = "Strong"
        elif score >= 50:
            health = "Moderate"
        else:
            health = "Weak"

        return {
            "score": score,
            "health": health,
            "factors": factors,
        }

    def _fetch_market_research(self, ticker: str, state_research: dict[str, Any]) -> dict[str, Any]:
        """Get market research - prefer data from Data Scout, fallback to direct fetch."""
        research_data = {
            "financial_analysis": "",
            "competitor_analysis": "",
            "risk_factors": "",
            "has_research": False,
        }
        
        # First, try to use research from Data Scout
        if state_research.get("research_available"):
            research_data["financial_analysis"] = state_research.get("fundamental_research", "")
            research_data["competitor_analysis"] = state_research.get("competitor_analysis", "")
            research_data["risk_factors"] = state_research.get("risk_factors", "")
            research_data["has_research"] = bool(research_data["financial_analysis"] or research_data["competitor_analysis"])
            if research_data["has_research"]:
                return research_data
        
        # Fallback to direct fetch if Data Scout research not available
        try:
            result = search_financial_analysis.invoke({"ticker": ticker, "topic": "earnings valuation"})
            if result and "not configured" not in result.lower():
                research_data["financial_analysis"] = result
                research_data["has_research"] = True
        except Exception:
            pass
        
        try:
            result = search_competitor_analysis.invoke({"ticker": ticker})
            if result and "not configured" not in result.lower():
                research_data["competitor_analysis"] = result
                research_data["has_research"] = True
        except Exception:
            pass
        
        try:
            result = search_risk_factors.invoke({"ticker": ticker})
            if result and "not configured" not in result.lower():
                research_data["risk_factors"] = result
                research_data["has_research"] = True
        except Exception:
            pass
        
        return research_data

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        """Perform fundamental analysis on company data."""
        from datetime import datetime
        
        ticker = state.get("ticker", "")
        company_info = state.get("company_info", {})
        financials = state.get("financials", {})
        current_date = datetime.now().strftime("%B %d, %Y")

        if not company_info and not ticker:
            return {
                **state,
                "fundamental_analysis": {"error": "No company data available"},
                "fundamental_signal": {"signal": "Neutral", "confidence": 0.0},
            }

        # Get market research (from Data Scout or direct fetch)
        market_research = state.get("market_research", {})
        research_data = self._fetch_market_research(ticker, market_research)
        research_timestamp = market_research.get("fetch_timestamp", "recently")

        ratios = self.calculate_ratios(company_info, financials)
        health_assessment = self._assess_financial_health(ratios)

        if health_assessment["health"] == "Strong":
            signal = "Bullish"
            confidence = 0.7
        elif health_assessment["health"] == "Weak":
            signal = "Bearish"
            confidence = 0.6
        else:
            signal = "Neutral"
            confidence = 0.5

        # Build comprehensive analysis prompt with market research
        research_context = ""
        if research_data["has_research"]:
            research_context = f"""
## ⚠️ REAL-TIME MARKET RESEARCH (Fetched: {research_timestamp}) - USE THIS AS PRIMARY SOURCE:

### Financial Analysis from Web Sources:
{research_data['financial_analysis'][:1500] if research_data['financial_analysis'] else 'No data available'}

### Competitive Landscape:
{research_data['competitor_analysis'][:1500] if research_data['competitor_analysis'] else 'No data available'}

### Key Risk Factors:
{research_data['risk_factors'][:1500] if research_data['risk_factors'] else 'No data available'}
"""
        else:
            research_context = "\n## ⚠️ No real-time market research available. Use provided financial data only.\n"

        analysis_prompt = f"""TODAY'S DATE: {current_date}

Provide a comprehensive fundamental analysis for {ticker}:

## Company Information:
{company_info}

## Financial Ratios:
{ratios}

## Health Assessment:
{health_assessment}

{research_context}

⚠️ CRITICAL INSTRUCTIONS:
- Today is {current_date}. Do NOT reference outdated information.
- The "REAL-TIME MARKET RESEARCH" section above contains data fetched TODAY from the web.
- You MUST use this real-time data as your PRIMARY source for recent developments.
- DO NOT rely on your training data for recent earnings, news, or market conditions.

Please provide a detailed analysis including:

1. **Financial Health Assessment** - concise summary of strength/stability
2. **Valuation Analysis** - fair/over/under valued? (Use the real-time market research)
3. **Competitive Position** - key peers comparison (from real-time research)
4. **Key Strengths** - competitive advantages
5. **Key Weaknesses/Risks** - main concerns (from real-time risk factors)
6. **Long-term Investment Thesis** - fundamental recommendation

Be CONCISE. Focus on the most critical insights. Reference the real-time data."""

        fundamental_summary = self.invoke(analysis_prompt)

        return {
            **state,
            "financial_ratios": ratios,
            "health_assessment": health_assessment,
            "fundamental_signal": {"signal": signal, "confidence": confidence},
            "fundamental_analysis": fundamental_summary,
        }
