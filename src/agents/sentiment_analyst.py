
from typing import Any, Optional

from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.data.tavily_tools import search_stock_news, search_market_sentiment, get_tools_for_agent


class SentimentAnalystAgent(BaseAgent):
    """Agent responsible for sentiment analysis of market news and social signals."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.3,
    ):
        super().__init__(ollama_client, model, temperature)

    @property
    def name(self) -> str:
        return "sentiment_analyst"

    @property
    def role(self) -> str:
        return "Sentiment Analyst - Market Sentiment and News Specialist"

    @property
    def system_prompt(self) -> str:
        return """You are the Sentiment Analyst agent in a hedge fund analysis team.

CRITICAL: You will be provided with REAL-TIME news and sentiment data fetched from the web.
You MUST use this provided data as your PRIMARY source of information.
DO NOT rely on your training data for recent news, events, or market sentiment.

Your responsibilities:
1. Analyze news headlines and articles for sentiment
2. Gauge retail and institutional sentiment
3. Identify sentiment trends and shifts
4. Detect potential catalysts from news flow
5. Use the PROVIDED news and sentiment sources for current context

Your analysis should include:
- Overall sentiment score (-1.0 to +1.0)
- Sentiment trend (Improving/Stable/Declining)
- Key news themes and their impact FROM THE PROVIDED NEWS
- Potential catalysts (positive and negative)

Important considerations:
- Distinguish between noise and signal
- Weight recent news more heavily
- Consider source credibility
- Note any conflicting signals

Output format:
- Sentiment score with confidence
- Key themes summary from REAL-TIME news
- Catalyst identification
- Sentiment signal (Bullish/Bearish/Neutral)
"""

    def analyze_text_sentiment(self, text: str) -> dict[str, Any]:
        """Analyze sentiment of a text using the LLM."""
        prompt = f"""Analyze the sentiment of the following text and provide a JSON-like response:

Text: {text}

Respond with:
- sentiment_score: float between -1.0 (very negative) and 1.0 (very positive)
- confidence: float between 0.0 and 1.0
- key_themes: list of main themes
- brief_analysis: one sentence summary"""

        response = self.invoke(prompt)
        return {"raw_analysis": response, "text_analyzed": text[:200]}

    def aggregate_sentiment(
        self,
        company_info: dict[str, Any],
        news_data: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Aggregate sentiment from multiple sources."""
        ticker = company_info.get("name", "Unknown")
        sector = company_info.get("sector", "Unknown")
        description = company_info.get("description", "")

        context_prompt = f"""Based on the following company information, provide a sentiment analysis:

Company: {ticker}
Sector: {sector}
Description: {description[:500] if description else 'N/A'}

Consider:
1. General market sentiment for this sector
2. Company-specific factors from the description
3. Any implied risks or opportunities

Provide:
- sentiment_score: float between -1.0 and 1.0
- confidence: float between 0.0 and 1.0
- key_factors: list of factors influencing sentiment
- outlook: brief outlook statement"""

        analysis = self.invoke(context_prompt)

        return {
            "aggregated_analysis": analysis,
            "sources_analyzed": ["company_description", "sector_context"],
        }

    def _extract_sentiment_score(self, analysis: str) -> float:
        """Extract numerical sentiment score from analysis text."""
        import re

        patterns = [
            r"sentiment[_\s]?score[:\s]+([+-]?\d*\.?\d+)",
            r"score[:\s]+([+-]?\d*\.?\d+)",
            r"([+-]?\d*\.?\d+)\s*(?:out of|/)",
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                try:
                    score = float(match.group(1))
                    if -1.0 <= score <= 1.0:
                        return score
                    elif 0 <= score <= 10:
                        return (score - 5) / 5
                except ValueError:
                    continue

        positive_words = ["bullish", "positive", "optimistic", "strong", "growth", "opportunity"]
        negative_words = ["bearish", "negative", "pessimistic", "weak", "decline", "risk", "concern"]

        analysis_lower = analysis.lower()
        pos_count = sum(1 for word in positive_words if word in analysis_lower)
        neg_count = sum(1 for word in negative_words if word in analysis_lower)

        if pos_count + neg_count > 0:
            return (pos_count - neg_count) / (pos_count + neg_count)

        return 0.0

    def _fetch_market_intelligence(self, ticker: str, state_research: dict[str, Any]) -> dict[str, Any]:
        """Get market intelligence - prefer data from Data Scout, fallback to direct fetch."""
        intel_data = {
            "news": "",
            "sentiment_sources": "",
            "has_intel": False,
        }
        
        # First, try to use research from Data Scout
        if state_research.get("research_available"):
            intel_data["sentiment_sources"] = state_research.get("sentiment_sources", "")
            intel_data["has_intel"] = bool(intel_data["sentiment_sources"])
        
        # Also fetch fresh news
        try:
            news_result = search_stock_news.invoke({"ticker": ticker, "query": ""})
            if news_result and "not configured" not in news_result.lower():
                intel_data["news"] = news_result
                intel_data["has_intel"] = True
        except Exception:
            pass
        
        try:
            if not intel_data["sentiment_sources"]:
                sentiment_result = search_market_sentiment.invoke({"ticker": ticker})
                if sentiment_result and "not configured" not in sentiment_result.lower():
                    intel_data["sentiment_sources"] = sentiment_result
                    intel_data["has_intel"] = True
        except Exception:
            pass
        
        return intel_data

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        """Perform sentiment analysis on available data."""
        ticker = state.get("ticker", "")
        company_info = state.get("company_info", {})
        news_data = state.get("news_data", [])

        if not company_info and not ticker:
            return {
                **state,
                "sentiment_analysis": {"error": "No data available for sentiment analysis"},
                "sentiment_signal": {"signal": "Neutral", "confidence": 0.0, "score": 0.0},
            }

        # Get market intelligence (from Data Scout or direct fetch)
        market_research = state.get("market_research", {})
        intel_data = self._fetch_market_intelligence(ticker, market_research)
        
        aggregated = self.aggregate_sentiment(company_info, news_data)
        sentiment_score = self._extract_sentiment_score(aggregated.get("aggregated_analysis", ""))

        if sentiment_score > 0.2:
            signal = "Bullish"
            confidence = min(0.8, 0.5 + abs(sentiment_score))
        elif sentiment_score < -0.2:
            signal = "Bearish"
            confidence = min(0.8, 0.5 + abs(sentiment_score))
        else:
            signal = "Neutral"
            confidence = 0.5

        # Build comprehensive analysis prompt with market intelligence
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        research_timestamp = market_research.get("fetch_timestamp", "recently")
        
        intel_context = ""
        if intel_data["has_intel"]:
            intel_context = f"""
## ⚠️ REAL-TIME NEWS & MARKET INTELLIGENCE (Fetched: {research_timestamp}) - USE THIS AS PRIMARY SOURCE:

### Latest News Headlines & Articles:
{intel_data['news'][:2000] if intel_data['news'] else 'No recent news available'}

### Investor Sentiment Sources (Reddit, Twitter, StockTwits):
{intel_data['sentiment_sources'][:2000] if intel_data['sentiment_sources'] else 'No sentiment sources available'}
"""
        else:
            intel_context = "\n## ⚠️ No real-time news available. Use provided company data only.\n"

        analysis_prompt = f"""TODAY'S DATE: {current_date}

Provide a comprehensive sentiment analysis for {ticker}:

## Company Context:
{aggregated}

{intel_context}

## Preliminary Sentiment Score: {sentiment_score}

⚠️ CRITICAL INSTRUCTIONS:
- Today is {current_date}. Do NOT reference outdated news or events.
- The "REAL-TIME NEWS & MARKET INTELLIGENCE" section above contains data fetched TODAY.
- You MUST use this real-time news as your PRIMARY source for sentiment analysis.
- DO NOT rely on your training data for recent news, events, or market sentiment.
- Reference specific news items from the provided data in your analysis.

Please provide a detailed analysis including:

1. **Overall Market Sentiment** - Bullish/Bearish/Neutral with confidence level
2. **Key Sentiment Drivers** - What factors are driving current sentiment (cite real-time news)
3. **Recent News Impact** - How the PROVIDED recent news is affecting investor perception
4. **Potential Catalysts to Watch**
   - Positive catalysts that could drive the stock higher
   - Negative catalysts or risks to monitor
5. **Sentiment-Based Recommendation** - Your recommendation based purely on the real-time sentiment data"""

        sentiment_summary = self.invoke(analysis_prompt)

        return {
            **state,
            "sentiment_score": sentiment_score,
            "sentiment_aggregation": aggregated,
            "market_sentiment_data": intel_data if intel_data["has_intel"] else {},
            "sentiment_signal": {
                "signal": signal,
                "confidence": round(confidence, 2),
                "score": round(sentiment_score, 2),
            },
            "sentiment_analysis": sentiment_summary,
        }
