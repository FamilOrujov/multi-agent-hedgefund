
from typing import Any, Optional
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from src.config.settings import get_settings
from src.data.tavily_client import TavilyClient


def get_tavily_search_tool(max_results: int = 5) -> Optional[TavilySearchResults]:
    """
    Get the Tavily search tool for use with LangChain agents.
    
    Returns None if Tavily API key is not configured.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        return None
    
    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
    )


# Create tool instances using the @tool decorator for custom tools
_tavily_client: Optional[TavilyClient] = None


def _get_client() -> TavilyClient:
    """Get or create the Tavily client singleton."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


@tool
def search_stock_news(ticker: str, query: str = "") -> str:
    """
    Search for recent news articles about a stock ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, MSFT)
        query: Optional additional search terms to refine results
        
    Returns:
        A formatted string containing recent news headlines and summaries
    """
    client = _get_client()
    if not client.is_configured:
        return "Tavily API is not configured. Please set TAVILY_API_KEY in environment."
    
    results = client.search_news(ticker, query, max_results=5)
    
    if not results:
        return f"No recent news found for {ticker}"
    
    if results and "error" in results[0]:
        return f"Error searching news: {results[0]['error']}"
    
    output = f"## Recent News for {ticker}\n\n"
    for i, article in enumerate(results, 1):
        title = article.get("title", "No title")
        content = article.get("content", "")[:300]
        url = article.get("url", "")
        output += f"**{i}. {title}**\n"
        output += f"{content}...\n"
        output += f"Source: {url}\n\n"
    
    return output


@tool
def search_financial_analysis(ticker: str, topic: str = "financial analysis") -> str:
    """
    Search for in-depth financial analysis on a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, MSFT)
        topic: Analysis topic like 'earnings', 'valuation', 'risks', 'growth prospects'
        
    Returns:
        A formatted string containing analysis summaries from financial sources
    """
    client = _get_client()
    if not client.is_configured:
        return "Tavily API is not configured. Please set TAVILY_API_KEY in environment."
    
    results = client.search_analysis(ticker, topic, max_results=5)
    
    if not results:
        return f"No analysis found for {ticker} on topic: {topic}"
    
    if results and "error" in results[0]:
        return f"Error searching analysis: {results[0]['error']}"
    
    output = f"## {topic.title()} Analysis for {ticker}\n\n"
    for i, article in enumerate(results, 1):
        title = article.get("title", "No title")
        content = article.get("content", "")[:400]
        output += f"**{i}. {title}**\n"
        output += f"{content}...\n\n"
    
    return output


@tool
def search_market_sentiment(ticker: str) -> str:
    """
    Search for market sentiment and investor opinions about a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, MSFT)
        
    Returns:
        A formatted string containing sentiment indicators from various sources
    """
    client = _get_client()
    if not client.is_configured:
        return "Tavily API is not configured. Please set TAVILY_API_KEY in environment."
    
    results = client.search_sentiment_sources(ticker, max_results=5)
    
    if not results:
        return f"No sentiment data found for {ticker}"
    
    if results and "error" in results[0]:
        return f"Error searching sentiment: {results[0]['error']}"
    
    output = f"## Market Sentiment for {ticker}\n\n"
    for i, source in enumerate(results, 1):
        title = source.get("title", "No title")
        content = source.get("content", "")[:350]
        url = source.get("url", "")
        output += f"**{i}. {title}**\n"
        output += f"{content}...\n"
        output += f"Source: {url}\n\n"
    
    return output


@tool
def search_company_overview(ticker: str) -> str:
    """
    Get comprehensive company information and business overview.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, MSFT)
        
    Returns:
        A formatted string containing company overview and business model information
    """
    client = _get_client()
    if not client.is_configured:
        return "Tavily API is not configured. Please set TAVILY_API_KEY in environment."
    
    result = client.get_company_info(ticker)
    
    if not result:
        return f"No company information found for {ticker}"
    
    if "error" in result:
        return f"Error getting company info: {result['error']}"
    
    output = f"## Company Overview: {ticker}\n\n"
    output += result.get("content", "No content available")
    output += f"\n\nSources: {len(result.get('urls', []))} web sources analyzed"
    
    return output


@tool
def search_competitor_analysis(ticker: str) -> str:
    """
    Search for competitor analysis and industry comparison for a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, MSFT)
        
    Returns:
        A formatted string containing competitor and industry analysis
    """
    client = _get_client()
    if not client.is_configured:
        return "Tavily API is not configured. Please set TAVILY_API_KEY in environment."
    
    results = client.search_analysis(ticker, "competitors industry comparison market share", max_results=5)
    
    if not results:
        return f"No competitor analysis found for {ticker}"
    
    if results and "error" in results[0]:
        return f"Error searching competitor analysis: {results[0]['error']}"
    
    output = f"## Competitor Analysis for {ticker}\n\n"
    for i, article in enumerate(results, 1):
        title = article.get("title", "No title")
        content = article.get("content", "")[:400]
        output += f"**{i}. {title}**\n"
        output += f"{content}...\n\n"
    
    return output


@tool
def search_risk_factors(ticker: str) -> str:
    """
    Search for risk factors and potential concerns about a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, MSFT)
        
    Returns:
        A formatted string containing identified risk factors
    """
    client = _get_client()
    if not client.is_configured:
        return "Tavily API is not configured. Please set TAVILY_API_KEY in environment."
    
    results = client.search_analysis(ticker, "risks concerns challenges headwinds", max_results=5)
    
    if not results:
        return f"No risk analysis found for {ticker}"
    
    if results and "error" in results[0]:
        return f"Error searching risks: {results[0]['error']}"
    
    output = f"## Risk Factors for {ticker}\n\n"
    for i, article in enumerate(results, 1):
        title = article.get("title", "No title")
        content = article.get("content", "")[:400]
        output += f"**{i}. {title}**\n"
        output += f"{content}...\n\n"
    
    return output


def get_all_tavily_tools() -> list:
    """
    Get all available Tavily tools for use with agents.
    
    Returns:
        List of tool functions that can be bound to LangChain agents
    """
    client = _get_client()
    if not client.is_configured:
        return []
    
    return [
        search_stock_news,
        search_financial_analysis,
        search_market_sentiment,
        search_company_overview,
        search_competitor_analysis,
        search_risk_factors,
    ]


def get_tools_for_agent(agent_type: str) -> list:
    """
    Get relevant Tavily tools based on agent type.
    
    Args:
        agent_type: One of 'data_scout', 'technical', 'fundamental', 'sentiment', 'portfolio_manager'
        
    Returns:
        List of relevant tools for the agent type
    """
    client = _get_client()
    if not client.is_configured:
        return []
    
    tool_mapping = {
        "data_scout": [
            search_stock_news,
            search_company_overview,
        ],
        "technical": [
            search_stock_news,
            search_financial_analysis,
        ],
        "fundamental": [
            search_financial_analysis,
            search_company_overview,
            search_competitor_analysis,
            search_risk_factors,
        ],
        "sentiment": [
            search_stock_news,
            search_market_sentiment,
        ],
        "portfolio_manager": [
            search_stock_news,
            search_risk_factors,
        ],
    }
    
    return tool_mapping.get(agent_type, [])
