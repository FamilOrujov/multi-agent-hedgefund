
from fastapi import APIRouter, Depends

from src.api.schemas import HealthResponse, OllamaHealthResponse, TavilyHealthResponse
from src.api.dependencies import get_ollama_client, get_tavily_client
from src.llm.ollama_client import OllamaClient
from src.data.tavily_client import TavilyClient

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(status="healthy")


@router.get("/ollama", response_model=OllamaHealthResponse)
async def ollama_health(client: OllamaClient = Depends(get_ollama_client)):
    """Check Ollama LLM connection status."""
    try:
        # Try to get available models
        is_available = client.is_available()
        model = client.model if is_available else None
        return OllamaHealthResponse(
            connected=is_available,
            model=model,
        )
    except Exception as e:
        return OllamaHealthResponse(
            connected=False,
            error=str(e),
        )


@router.get("/tavily", response_model=TavilyHealthResponse)
async def tavily_health(client: TavilyClient = Depends(get_tavily_client)):
    """Check Tavily API configuration status."""
    try:
        return TavilyHealthResponse(
            configured=client.is_configured,
        )
    except Exception as e:
        return TavilyHealthResponse(
            configured=False,
            error=str(e),
        )
