
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import AgentInfo, AgentListResponse, AgentTestRequest, AgentTestResponse
from src.api.dependencies import get_ollama_client, get_agent_by_name, get_all_agents
from src.llm.ollama_client import OllamaClient

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.get("/", response_model=AgentListResponse)
async def list_agents(client: OllamaClient = Depends(get_ollama_client)):
    """List all available agents with their configuration."""
    agents = get_all_agents(client)
    
    agent_infos = []
    for name, agent in agents.items():
        agent_infos.append(AgentInfo(
            name=agent.name,
            role=agent.role,
            model=agent.model,
            temperature=agent.temperature,
        ))
    
    return AgentListResponse(agents=agent_infos, count=len(agent_infos))


@router.get("/{name}", response_model=AgentInfo)
async def get_agent(name: str, client: OllamaClient = Depends(get_ollama_client)):
    """Get information about a specific agent."""
    agent = get_agent_by_name(name, client)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    
    return AgentInfo(
        name=agent.name,
        role=agent.role,
        model=agent.model,
        temperature=agent.temperature,
    )


@router.post("/{name}/test", response_model=AgentTestResponse)
async def test_agent(
    name: str,
    request: AgentTestRequest,
    client: OllamaClient = Depends(get_ollama_client),
):
    """
    Test a specific agent with a ticker.
    
    This runs the agent's analyze() method and returns the results.
    Useful for debugging individual agents.
    """
    agent = get_agent_by_name(name, client)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    
    # Build initial state
    state = {"ticker": request.ticker}
    if request.custom_state:
        state.update(request.custom_state)
    
    # For non-data-scout agents, we need data from data scout first
    if name != "data_scout" and not request.custom_state:
        data_scout = get_agent_by_name("data_scout", client)
        if data_scout:
            state = data_scout.analyze(state)
    
    start_time = time.time()
    try:
        result = agent.analyze(state)
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Extract signal based on agent type
        signal = None
        if name == "technical_analyst":
            signal = result.get("technical_signal")
        elif name == "fundamental_analyst":
            signal = result.get("fundamental_signal")
        elif name == "sentiment_analyst":
            signal = result.get("sentiment_signal")
        elif name == "portfolio_manager":
            signal = result.get("manager_decision")
        
        # Extract analysis
        analysis = None
        if name == "data_scout":
            analysis = result.get("data_scout_summary")
        elif name == "technical_analyst":
            analysis = result.get("technical_analysis")
        elif name == "fundamental_analyst":
            analysis = result.get("fundamental_analysis")
        elif name == "sentiment_analyst":
            analysis = result.get("sentiment_analysis")
        elif name == "portfolio_manager":
            analysis = result.get("final_thesis")
        
        return AgentTestResponse(
            agent=name,
            ticker=request.ticker,
            analysis=analysis,
            signal=signal,
            duration_ms=duration_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent test failed: {str(e)}")


@router.get("/{name}/config")
async def get_agent_config(name: str, client: OllamaClient = Depends(get_ollama_client)):
    """Get the configuration and system prompt of an agent."""
    agent = get_agent_by_name(name, client)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    
    return {
        "name": agent.name,
        "role": agent.role,
        "model": agent.model,
        "temperature": agent.temperature,
        "system_prompt": agent.system_prompt,
    }
