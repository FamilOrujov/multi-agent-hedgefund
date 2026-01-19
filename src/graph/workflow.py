
from typing import Any, Literal, Optional, AsyncIterator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.data_scout import DataScoutAgent
from src.agents.technical_analyst import TechnicalAnalystAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.sentiment_analyst import SentimentAnalystAgent
from src.agents.portfolio_manager import PortfolioManagerAgent
from src.llm.ollama_client import OllamaClient
from src.config.settings import get_settings
from src.graph.workflow_streaming import run_analysis_streaming
from src.graph.guardrails import (
    HITLManager,
    GuardrailConfig,
    create_guardrail_node,
    create_human_review_node,
)

def create_analysis_graph(
    ollama_client: Optional[OllamaClient] = None,
    llm_model: Optional[str] = None,
    guardrail_config: Optional[GuardrailConfig] = None,
    enable_hitl: bool = True,
) -> tuple[StateGraph, HITLManager]:
    """
    Create the LangGraph workflow for hedge fund analysis.
    
    The graph implements a cyclic reasoning pattern:
    DataScout -> [Technical, Fundamental, Sentiment] -> PortfolioManager -> Guardrails -> (HITL or end)
    
    Args:
        ollama_client: Optional OllamaClient instance
        llm_model: Optional specific model to use
        guardrail_config: Configuration for guardrails
        enable_hitl: Whether to enable human-in-the-loop checkpoints
        
    Returns:
        Tuple of (StateGraph, HITLManager)
    """
    settings = get_settings()
    client = ollama_client or OllamaClient()
    
    # Initialize HITL manager with guardrails
    hitl_manager = HITLManager(guardrail_config)

    data_scout = DataScoutAgent(client, model=llm_model)
    technical_analyst = TechnicalAnalystAgent(client, model=llm_model)
    fundamental_analyst = FundamentalAnalystAgent(client, model=llm_model)
    sentiment_analyst = SentimentAnalystAgent(client, model=llm_model)
    portfolio_manager = PortfolioManagerAgent(client, model=llm_model)

    def data_scout_node(state: dict[str, Any]) -> dict[str, Any]:
        """Fetch all raw data for analysis."""
        state["current_agent"] = "data_scout"
        state["status"] = "running"
        return data_scout.analyze(state)

    def technical_node(state: dict[str, Any]) -> dict[str, Any]:
        """Perform technical analysis."""
        state["current_agent"] = "technical_analyst"
        return technical_analyst.analyze(state)

    def fundamental_node(state: dict[str, Any]) -> dict[str, Any]:
        """Perform fundamental analysis."""
        state["current_agent"] = "fundamental_analyst"
        return fundamental_analyst.analyze(state)

    def sentiment_node(state: dict[str, Any]) -> dict[str, Any]:
        """Perform sentiment analysis."""
        state["current_agent"] = "sentiment_analyst"
        return sentiment_analyst.analyze(state)

    def manager_node(state: dict[str, Any]) -> dict[str, Any]:
        """Synthesize analyses and make decision."""
        state["current_agent"] = "portfolio_manager"
        return portfolio_manager.analyze(state)

    def guardrails_node(state: dict[str, Any]) -> dict[str, Any]:
        """Apply guardrails and determine if human review is needed."""
        state["current_agent"] = "guardrails"
        return hitl_manager.evaluate_state(state)

    def should_continue(state: dict[str, Any]) -> Literal["human_review", "refine", "end"]:
        """Determine next step after guardrails evaluation."""
        manager_decision = state.get("manager_decision", {})
        confidence = manager_decision.get("confidence", 0)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", settings.max_iterations)
        requires_review = state.get("requires_human_review", False)

        # Max iterations reached - go to finalize
        if iteration_count >= max_iterations:
            return "end"

        # Guardrails triggered human review
        if requires_review and enable_hitl:
            return "human_review"

        # High confidence - auto-approve
        if confidence >= 75:
            return "end"

        # Medium confidence - may need refinement
        if confidence >= 50:
            return "end"

        # Low confidence - refine
        return "refine"

    def human_review_node(state: dict[str, Any]) -> dict[str, Any]:
        """Prepare state for human review (interrupt point)."""
        review_request = hitl_manager.create_review_request(state)
        state["pending_review_id"] = review_request["review_id"]
        state["status"] = "awaiting_review"
        state["review_request"] = review_request
        return state

    def refine_node(state: dict[str, Any]) -> dict[str, Any]:
        """Prepare for another iteration of analysis."""
        feedback = portfolio_manager.get_critique_feedback(state)
        state["human_feedback"] = feedback
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        return state

    def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
        """Finalize the analysis."""
        state["status"] = "completed"
        state["current_agent"] = "finalize"
        return state

    graph = StateGraph(dict)

    # Add all nodes
    graph.add_node("data_scout", data_scout_node)
    graph.add_node("technical_analyst", technical_node)
    graph.add_node("fundamental_analyst", fundamental_node)
    graph.add_node("sentiment_analyst", sentiment_node)
    graph.add_node("portfolio_manager", manager_node)
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("refine", refine_node)
    graph.add_node("finalize", finalize_node)

    # Set entry point
    graph.set_entry_point("data_scout")

    # Define edges - sequential analysis flow
    graph.add_edge("data_scout", "technical_analyst")
    graph.add_edge("technical_analyst", "fundamental_analyst")
    graph.add_edge("fundamental_analyst", "sentiment_analyst")
    graph.add_edge("sentiment_analyst", "portfolio_manager")
    graph.add_edge("portfolio_manager", "guardrails")

    # Conditional edges after guardrails
    graph.add_conditional_edges(
        "guardrails",
        should_continue,
        {
            "human_review": "human_review",
            "refine": "refine",
            "end": "finalize",
        },
    )

    # Human review leads to finalize
    graph.add_edge("human_review", "finalize")

    # Refine loops back to technical analyst
    graph.add_edge("refine", "technical_analyst")

    # Finalize ends the graph
    graph.add_edge("finalize", END)

    return graph, hitl_manager


def run_analysis(
    ticker: str,
    ollama_client: Optional[OllamaClient] = None,
    llm_model: Optional[str] = None,
    analysis_depth: str = "standard",
    guardrail_config: Optional[GuardrailConfig] = None,
    enable_hitl: bool = True,
) -> tuple[dict[str, Any], HITLManager]:
    """
    Run a complete analysis for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        ollama_client: Optional OllamaClient instance
        llm_model: Optional specific model to use
        analysis_depth: quick/standard/deep
        guardrail_config: Configuration for guardrails
        enable_hitl: Whether to enable human-in-the-loop
        
    Returns:
        Tuple of (final state dictionary, HITLManager instance)
    """
    settings = get_settings()

    graph, hitl_manager = create_analysis_graph(
        ollama_client, 
        llm_model,
        guardrail_config,
        enable_hitl,
    )
    
    # Compile with memory saver for checkpointing if HITL is enabled
    if enable_hitl:
        memory = MemorySaver()
        compiled_graph = graph.compile(checkpointer=memory)
    else:
        compiled_graph = graph.compile()

    initial_state = {
        "ticker": ticker.upper(),
        "analysis_depth": analysis_depth,
        "iteration_count": 0,
        "max_iterations": settings.max_iterations,
        "status": "pending",
        "errors": [],
        "price_data": {},
        "price_summary": {},
        "company_info": {},
        "financials": {},
        "news_data": [],
        "technical_indicators": {},
        "technical_signal": {},
        "financial_ratios": {},
        "health_assessment": {},
        "fundamental_signal": {},
        "sentiment_score": 0.0,
        "sentiment_aggregation": {},
        "sentiment_signal": {},
        "signal_aggregation": {},
        "manager_decision": {},
        "requires_human_review": False,
        "review_triggers": [],
        "guardrail_results": [],
    }

    # For HITL, we need a thread_id for checkpointing
    config = {"configurable": {"thread_id": f"{ticker}_{settings.max_iterations}"}} if enable_hitl else {}
    
    final_state = compiled_graph.invoke(initial_state, config)

    return final_state, hitl_manager
