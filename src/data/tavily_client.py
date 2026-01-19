
from typing import Any, Optional
from datetime import datetime

from tavily import TavilyClient as BaseTavilyClient

from src.config.settings import get_settings


class TavilyClient:
    """
    Tavily API client for deep web search and news analysis.
    
    Tavily provides:
    - Real-time web search with AI-powered relevance
    - News article retrieval
    - Content extraction and summarization
    """

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.tavily_api_key
        
        if self.api_key:
            self._client = BaseTavilyClient(api_key=self.api_key)
        else:
            self._client = None

    @property
    def is_configured(self) -> bool:
        """Check if Tavily is configured with an API key."""
        return self._client is not None

    def search_news(
        self,
        ticker: str,
        query: Optional[str] = None,
        max_results: int = 10,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """
        Search for recent news about a ticker.
        
        Args:
            ticker: Stock ticker symbol
            query: Additional search query
            max_results: Maximum number of results
            days: Number of days to look back
            
        Returns:
            List of news articles with title, content, url, date
        """
        if not self.is_configured:
            return []

        search_query = f"{ticker} stock news"
        if query:
            search_query = f"{ticker} {query}"

        try:
            response = self._client.search(
                query=search_query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=["reuters.com", "bloomberg.com", "cnbc.com", 
                                "wsj.com", "marketwatch.com", "seekingalpha.com",
                                "finance.yahoo.com", "fool.com", "investopedia.com"],
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0),
                    "published_date": item.get("published_date", ""),
                })

            return results

        except Exception as e:
            return [{"error": str(e)}]

    def search_analysis(
        self,
        ticker: str,
        topic: str = "financial analysis",
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for in-depth analysis on a specific topic.
        
        Args:
            ticker: Stock ticker symbol
            topic: Analysis topic (e.g., "earnings", "valuation", "risks")
            max_results: Maximum number of results
            
        Returns:
            List of analysis articles
        """
        if not self.is_configured:
            return []

        current_year = datetime.now().year
        search_query = f"{ticker} {topic} analysis {current_year} latest"

        try:
            response = self._client.search(
                query=search_query,
                search_depth="advanced",
                max_results=max_results,
            )

            return [
                {
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0),
                }
                for item in response.get("results", [])
            ]

        except Exception as e:
            return [{"error": str(e)}]

    def get_company_info(self, ticker: str) -> dict[str, Any]:
        """
        Get comprehensive company information from web sources.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        if not self.is_configured:
            return {}

        try:
            response = self._client.search(
                query=f"{ticker} company overview business model",
                search_depth="advanced",
                max_results=3,
            )

            contents = [r.get("content", "") for r in response.get("results", [])]
            
            return {
                "sources": len(contents),
                "content": "\n\n".join(contents),
                "urls": [r.get("url", "") for r in response.get("results", [])],
            }

        except Exception as e:
            return {"error": str(e)}

    def search_sentiment_sources(
        self,
        ticker: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for sentiment-relevant sources (social media, forums, etc.).
        
        Args:
            ticker: Stock ticker symbol
            max_results: Maximum number of results
            
        Returns:
            List of sentiment sources
        """
        if not self.is_configured:
            return []

        try:
            response = self._client.search(
                query=f"{ticker} stock sentiment reddit twitter investor opinion",
                search_depth="advanced",
                max_results=max_results,
                include_domains=["reddit.com", "twitter.com", "stocktwits.com",
                                "seekingalpha.com", "fool.com"],
            )

            return [
                {
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0),
                }
                for item in response.get("results", [])
            ]

        except Exception as e:
            return [{"error": str(e)}]
