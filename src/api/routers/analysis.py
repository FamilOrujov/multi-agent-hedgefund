
import time
import uuid
import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    StreamEvent,
)
from src.api.dependencies import get_ollama_client
from src.llm.ollama_client import OllamaClient
from src.graph.workflow import run_analysis
from src.graph.guardrails import GuardrailConfig

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# In-memory storage for analysis results (use Redis/DB in production)
_analysis_cache: dict[str, AnalysisResponse] = {}


@router.post("/", response_model=AnalysisResponse)
async def start_analysis(
    request: AnalysisRequest,
    client: OllamaClient = Depends(get_ollama_client),
):
    """
    Start a full multi-agent stock analysis.
    
    This runs the complete analysis pipeline:
    DataScout -> Technical/Fundamental/Sentiment -> Portfolio Manager
    """
    analysis_id = str(uuid.uuid4())[:8]
    ticker = request.ticker.upper()
    
    # Create initial response
    response = AnalysisResponse(
        id=analysis_id,
        ticker=ticker,
        status=AnalysisStatus.RUNNING,
    )
    _analysis_cache[analysis_id] = response
    
    start_time = time.time()
    
    try:
        # Map depth to analysis_depth
        depth_map = {
            "quick": "quick",
            "standard": "standard",
            "deep": "comprehensive",
        }
        analysis_depth = depth_map.get(request.depth.value, "standard")
        
        # Configure guardrails
        guardrail_config = GuardrailConfig(
            enable_confidence_check=True,
            enable_consistency_check=True,
            min_confidence_threshold=60,
        )
        
        # Run analysis
        result = run_analysis(
            ticker=ticker,
            ollama_client=client,
            analysis_depth=analysis_depth,
            guardrail_config=guardrail_config,
            enable_hitl=request.enable_hitl,
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Extract results
        manager_decision = result.get("manager_decision", {})
        
        response = AnalysisResponse(
            id=analysis_id,
            ticker=ticker,
            status=AnalysisStatus.COMPLETED,
            decision=manager_decision.get("decision"),
            confidence=manager_decision.get("confidence"),
            position_size=manager_decision.get("position_size"),
            thesis=result.get("final_thesis"),
            technical_signal=result.get("technical_signal"),
            fundamental_signal=result.get("fundamental_signal"),
            sentiment_signal=result.get("sentiment_signal"),
            duration_ms=duration_ms,
        )
        _analysis_cache[analysis_id] = response
        
        return response
        
    except Exception as e:
        response = AnalysisResponse(
            id=analysis_id,
            ticker=ticker,
            status=AnalysisStatus.FAILED,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )
        _analysis_cache[analysis_id] = response
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    """Get the result of a previous analysis by ID."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _analysis_cache[analysis_id]


@router.websocket("/{ticker}/stream")
async def stream_analysis(websocket: WebSocket, ticker: str):
    """
    Stream analysis progress via WebSocket.
    
    Sends events as analysis progresses through each agent.
    """
    await websocket.accept()
    ticker = ticker.upper()
    
    try:
        # Import streaming workflow
        from src.graph.workflow_streaming import run_analysis_streaming
        from src.api.dependencies import get_ollama_client
        
        client = get_ollama_client()
        
        await websocket.send_json(StreamEvent(
            event="started",
            data={"ticker": ticker},
        ).model_dump(mode="json"))
        
        # Run streaming analysis
        async def stream_callback(event_data: dict):
            """Callback for streaming events."""
            event = StreamEvent(
                event=event_data.get("type", "update"),
                agent=event_data.get("agent"),
                data=event_data,
            )
            await websocket.send_json(event.model_dump(mode="json"))
        
        result = await run_analysis_streaming(
            ticker=ticker,
            ollama_client=client,
            stream_callback=stream_callback,
        )
        
        # Send final result
        manager_decision = result.get("manager_decision", {})
        await websocket.send_json(StreamEvent(
            event="completed",
            data={
                "decision": manager_decision.get("decision"),
                "confidence": manager_decision.get("confidence"),
                "thesis": result.get("final_thesis"),
            },
        ).model_dump(mode="json"))
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json(StreamEvent(
            event="error",
            data={"error": str(e)},
        ).model_dump(mode="json"))
    finally:
        await websocket.close()


@router.get("/")
async def list_analyses():
    """List all cached analyses."""
    return {
        "analyses": [
            {"id": aid, "ticker": a.ticker, "status": a.status.value}
            for aid, a in _analysis_cache.items()
        ],
        "count": len(_analysis_cache),
    }
