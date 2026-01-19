
from typing import Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AnalysisDepth(str, Enum):
    """Analysis depth options."""
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class AnalysisStatus(str, Enum):
    """Analysis status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# === Request Models ===

class AnalysisRequest(BaseModel):
    """Request to start a stock analysis."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, TSLA)")
    depth: AnalysisDepth = Field(default=AnalysisDepth.STANDARD, description="Analysis depth")
    enable_hitl: bool = Field(default=False, description="Enable human-in-the-loop review")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "depth": "standard",
                "enable_hitl": False
            }
        }


class AgentTestRequest(BaseModel):
    """Request to test a specific agent."""
    ticker: str = Field(..., description="Stock ticker to analyze")
    custom_state: Optional[dict[str, Any]] = Field(default=None, description="Custom state override")


# === Response Models ===

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="0.1.0")


class OllamaHealthResponse(BaseModel):
    """Ollama connection health."""
    connected: bool
    model: Optional[str] = None
    error: Optional[str] = None


class TavilyHealthResponse(BaseModel):
    """Tavily API health."""
    configured: bool
    error: Optional[str] = None


class AgentInfo(BaseModel):
    """Agent information."""
    name: str
    role: str
    model: Optional[str] = None
    temperature: Optional[float] = None


class AgentListResponse(BaseModel):
    """List of agents."""
    agents: list[AgentInfo]
    count: int


class AgentTestResponse(BaseModel):
    """Response from agent test."""
    agent: str
    ticker: str
    analysis: Any
    signal: Optional[dict[str, Any]] = None
    duration_ms: int


class AnalysisResponse(BaseModel):
    """Analysis result response."""
    id: str = Field(..., description="Analysis ID")
    ticker: str
    status: AnalysisStatus
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Results (populated when completed)
    decision: Optional[str] = None
    confidence: Optional[int] = None
    position_size: Optional[str] = None
    thesis: Optional[str] = None
    
    # Agent signals
    technical_signal: Optional[dict[str, Any]] = None
    fundamental_signal: Optional[dict[str, Any]] = None
    sentiment_signal: Optional[dict[str, Any]] = None
    
    # Metadata
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class StreamEvent(BaseModel):
    """Streaming event for WebSocket."""
    event: str = Field(..., description="Event type")
    agent: Optional[str] = None
    data: Any = None
    timestamp: datetime = Field(default_factory=datetime.now)
