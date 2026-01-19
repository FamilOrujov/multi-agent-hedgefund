
from typing import Optional
from functools import lru_cache

from src.llm.ollama_client import OllamaClient
from src.config.settings import Settings, get_settings
from src.data.tavily_client import TavilyClient


@lru_cache()
def get_ollama_client() -> OllamaClient:
    """Get cached OllamaClient instance."""
    return OllamaClient()


@lru_cache()
def get_tavily_client() -> TavilyClient:
    """Get cached TavilyClient instance."""
    return TavilyClient()


def get_settings_dep() -> Settings:
    """Get settings dependency."""
    return get_settings()


# Agent factory for testing
def get_agent_by_name(name: str, ollama_client: Optional[OllamaClient] = None):
    """Get an agent instance by name."""
    from src.agents.data_scout import DataScoutAgent
    from src.agents.technical_analyst import TechnicalAnalystAgent
    from src.agents.fundamental_analyst import FundamentalAnalystAgent
    from src.agents.sentiment_analyst import SentimentAnalystAgent
    from src.agents.portfolio_manager import PortfolioManagerAgent
    
    client = ollama_client or get_ollama_client()
    
    agents = {
        "data_scout": DataScoutAgent,
        "technical_analyst": TechnicalAnalystAgent,
        "fundamental_analyst": FundamentalAnalystAgent,
        "sentiment_analyst": SentimentAnalystAgent,
        "portfolio_manager": PortfolioManagerAgent,
    }
    
    agent_class = agents.get(name)
    if agent_class:
        return agent_class(client)
    return None


def get_all_agents(ollama_client: Optional[OllamaClient] = None) -> dict:
    """Get all agent instances."""
    from src.agents.data_scout import DataScoutAgent
    from src.agents.technical_analyst import TechnicalAnalystAgent
    from src.agents.fundamental_analyst import FundamentalAnalystAgent
    from src.agents.sentiment_analyst import SentimentAnalystAgent
    from src.agents.portfolio_manager import PortfolioManagerAgent
    
    client = ollama_client or get_ollama_client()
    
    return {
        "data_scout": DataScoutAgent(client),
        "technical_analyst": TechnicalAnalystAgent(client),
        "fundamental_analyst": FundamentalAnalystAgent(client),
        "sentiment_analyst": SentimentAnalystAgent(client),
        "portfolio_manager": PortfolioManagerAgent(client),
    }
