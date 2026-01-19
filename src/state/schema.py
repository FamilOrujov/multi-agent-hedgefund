
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class SignalData(BaseModel):
    """Signal data from an analyst."""

    signal: str = Field(default="Neutral", description="Bullish/Bearish/Neutral")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    score: Optional[float] = Field(default=None, description="Numerical score if applicable")


class ManagerDecision(BaseModel):
    """Portfolio manager's decision."""

    decision: str = Field(default="HOLD", description="BUY/SELL/HOLD")
    confidence: int = Field(default=0, ge=0, le=100)
    position_size: str = Field(default="None", description="Conservative/Moderate/Aggressive/None")
    has_consensus: bool = Field(default=False)


class AnalysisResult(BaseModel):
    """Final analysis result for persistence and reporting."""

    ticker: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Decision
    decision: str
    confidence: int
    position_size: str
    
    # Signals
    technical_signal: SignalData
    fundamental_signal: SignalData
    sentiment_signal: SignalData
    
    # Summaries
    technical_summary: str = ""
    fundamental_summary: str = ""
    sentiment_summary: str = ""
    final_thesis: str = ""
    
    # Metadata
    iterations: int = 0
    model_used: str = ""


class AgentState(BaseModel):
    """
    Shared state schema for the multi-agent system.
    
    This is the central state object passed between all agents in the LangGraph.
    It ensures type safety and prevents agents from inventing arbitrary fields.
    """

    # Input
    ticker: str = Field(..., description="Stock ticker symbol")
    analysis_depth: str = Field(default="standard", description="quick/standard/deep")
    
    # Data Scout outputs
    price_data: dict[str, Any] = Field(default_factory=dict)
    price_summary: dict[str, Any] = Field(default_factory=dict)
    company_info: dict[str, Any] = Field(default_factory=dict)
    financials: dict[str, Any] = Field(default_factory=dict)
    news_data: list[str] = Field(default_factory=list)
    data_scout_summary: str = ""
    
    # Technical Analyst outputs
    technical_indicators: dict[str, Any] = Field(default_factory=dict)
    technical_signal: dict[str, Any] = Field(default_factory=dict)
    technical_analysis: str = ""
    
    # Fundamental Analyst outputs
    financial_ratios: dict[str, Any] = Field(default_factory=dict)
    health_assessment: dict[str, Any] = Field(default_factory=dict)
    fundamental_signal: dict[str, Any] = Field(default_factory=dict)
    fundamental_analysis: str = ""
    
    # Sentiment Analyst outputs
    sentiment_score: float = 0.0
    sentiment_aggregation: dict[str, Any] = Field(default_factory=dict)
    sentiment_signal: dict[str, Any] = Field(default_factory=dict)
    sentiment_analysis: str = ""
    
    # Portfolio Manager outputs
    signal_aggregation: dict[str, Any] = Field(default_factory=dict)
    manager_decision: dict[str, Any] = Field(default_factory=dict)
    final_thesis: str = ""
    
    # Control flow
    iteration_count: int = Field(default=0, description="Current iteration in reasoning loop")
    max_iterations: int = Field(default=5, description="Maximum allowed iterations")
    current_agent: str = Field(default="", description="Currently active agent")
    status: str = Field(default="pending", description="pending/running/paused/completed/failed")
    
    # Human-in-the-loop
    human_feedback: Optional[str] = Field(default=None, description="Feedback from human review")
    requires_human_review: bool = Field(default=False)
    
    # Error handling
    errors: list[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True

    def to_langgraph_state(self) -> dict[str, Any]:
        """Convert to dictionary for LangGraph compatibility."""
        return self.model_dump()

    @classmethod
    def from_langgraph_state(cls, state: dict[str, Any]) -> "AgentState":
        """Create from LangGraph state dictionary."""
        return cls(**state)

    def to_result(self, model_used: str = "") -> AnalysisResult:
        """Convert current state to a final result."""
        return AnalysisResult(
            ticker=self.ticker,
            decision=self.manager_decision.get("decision", "HOLD"),
            confidence=self.manager_decision.get("confidence", 0),
            position_size=self.manager_decision.get("position_size", "None"),
            technical_signal=SignalData(**self.technical_signal) if self.technical_signal else SignalData(),
            fundamental_signal=SignalData(**self.fundamental_signal) if self.fundamental_signal else SignalData(),
            sentiment_signal=SignalData(
                signal=self.sentiment_signal.get("signal", "Neutral"),
                confidence=self.sentiment_signal.get("confidence", 0.0),
                score=self.sentiment_signal.get("score"),
            ) if self.sentiment_signal else SignalData(),
            technical_summary=self.technical_analysis,
            fundamental_summary=self.fundamental_analysis,
            sentiment_summary=self.sentiment_analysis,
            final_thesis=self.final_thesis,
            iterations=self.iteration_count,
            model_used=model_used,
        )
