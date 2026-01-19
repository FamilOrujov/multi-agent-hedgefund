
from typing import Any, AsyncIterator, Optional
import asyncio

from src.llm.ollama_client import OllamaClient
from src.agents.data_scout import DataScoutAgent
from src.agents.technical_analyst import TechnicalAnalystAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.sentiment_analyst import SentimentAnalystAgent
from src.agents.portfolio_manager import PortfolioManagerAgent
from src.config.settings import get_settings
from src.graph.guardrails import HITLManager, GuardrailConfig


async def run_analysis_streaming(
    ticker: str,
    ollama_client: Optional[OllamaClient] = None,
    llm_model: Optional[str] = None,
    indicators: Optional[dict] = None,
    analysis_depth: str = "standard",
    guardrail_config: Optional[GuardrailConfig] = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Run analysis with streaming events for TUI visualization.
    
    Yields events like:
    - {"type": "agent_start", "agent": "data_scout"}
    - {"type": "agent_thinking", "agent": "data_scout", "content": "..."}
    - {"type": "agent_done", "agent": "data_scout", "signal": "BUY", "confidence": 80}
    - {"type": "human_review", "question": "...", "context": "..."}
    - {"type": "final_decision", "decision": "BUY", "confidence": 85}
    - {"type": "complete", "result": {...}}
    """
    settings = get_settings()
    client = ollama_client or OllamaClient()
    hitl_manager = HITLManager(guardrail_config)
    
    data_scout = DataScoutAgent(client, model=llm_model)
    technical_analyst = TechnicalAnalystAgent(client, model=llm_model)
    fundamental_analyst = FundamentalAnalystAgent(client, model=llm_model)
    sentiment_analyst = SentimentAnalystAgent(client, model=llm_model)
    portfolio_manager = PortfolioManagerAgent(client, model=llm_model)
    
    state = {
        "ticker": ticker,
        "analysis_depth": analysis_depth,
        "indicators_config": indicators or {},
        "iteration": 0,
        "status": "starting",
    }
    
    yield {"type": "agent_start", "agent": "data_scout"}
    yield {"type": "agent_thinking", "agent": "data_scout", "content": f"Fetching market data for {ticker}..."}
    await asyncio.sleep(0.5)
    
    try:
        state = data_scout.analyze(state)
        yield {
            "type": "agent_thinking",
            "agent": "data_scout",
            "content": f"Retrieved price data, company info, and financials for {ticker}"
        }
        await asyncio.sleep(0.3)
        
        summary = state.get("data_scout_summary", "Data collection complete")[:200]
        yield {
            "type": "agent_done",
            "agent": "data_scout",
            "summary": summary,
        }
    except Exception as e:
        yield {"type": "agent_done", "agent": "data_scout", "summary": f"Error: {e}"}
    
    yield {"type": "agent_start", "agent": "technical_analyst"}
    yield {"type": "agent_thinking", "agent": "technical_analyst", "content": "Calculating technical indicators..."}
    await asyncio.sleep(0.5)
    
    try:
        state = technical_analyst.analyze(state)
        
        indicators_text = []
        tech_indicators = state.get("technical_indicators", {})
        if "rsi" in tech_indicators:
            rsi_val = tech_indicators["rsi"].get("value", 0)
            indicators_text.append(f"RSI: {rsi_val:.1f}")
        if "macd" in tech_indicators:
            macd_signal = tech_indicators["macd"].get("signal", "N/A")
            indicators_text.append(f"MACD: {macd_signal}")
        
        yield {
            "type": "agent_thinking",
            "agent": "technical_analyst",
            "content": "Indicators: " + ", ".join(indicators_text) if indicators_text else "Analysis complete"
        }
        await asyncio.sleep(0.3)
        
        tech_signal = state.get("technical_signal", {})
        yield {
            "type": "agent_done",
            "agent": "technical_analyst",
            "signal": tech_signal.get("signal", "HOLD"),
            "confidence": int(tech_signal.get("confidence", 0) * 100) if isinstance(tech_signal.get("confidence"), float) else tech_signal.get("confidence", 0),
            "reason": tech_signal.get("reasoning", ""),
            "summary": state.get("technical_analysis", "")[:150],
        }
    except Exception as e:
        yield {"type": "agent_done", "agent": "technical_analyst", "summary": f"Error: {e}"}
    
    yield {"type": "agent_start", "agent": "fundamental_analyst"}
    yield {"type": "agent_thinking", "agent": "fundamental_analyst", "content": "Analyzing financial statements and ratios..."}
    await asyncio.sleep(0.5)
    
    try:
        state = fundamental_analyst.analyze(state)
        
        health = state.get("health_assessment", {})
        health_score = health.get("score", 0)
        yield {
            "type": "agent_thinking",
            "agent": "fundamental_analyst",
            "content": f"Financial Health Score: {health_score}/100"
        }
        await asyncio.sleep(0.3)
        
        fund_signal = state.get("fundamental_signal", {})
        yield {
            "type": "agent_done",
            "agent": "fundamental_analyst",
            "signal": fund_signal.get("signal", "HOLD"),
            "confidence": fund_signal.get("confidence", 0),
            "reason": fund_signal.get("reasoning", ""),
            "summary": state.get("fundamental_analysis", "")[:150],
        }
    except Exception as e:
        yield {"type": "agent_done", "agent": "fundamental_analyst", "summary": f"Error: {e}"}
    
    yield {"type": "agent_start", "agent": "sentiment_analyst"}
    yield {"type": "agent_thinking", "agent": "sentiment_analyst", "content": "Analyzing market sentiment and news..."}
    await asyncio.sleep(0.5)
    
    try:
        state = sentiment_analyst.analyze(state)
        
        sentiment_score = state.get("sentiment_score", 0)
        sentiment_label = "Positive" if sentiment_score > 0.2 else ("Negative" if sentiment_score < -0.2 else "Neutral")
        yield {
            "type": "agent_thinking",
            "agent": "sentiment_analyst",
            "content": f"Sentiment: {sentiment_label} ({sentiment_score:.2f})"
        }
        await asyncio.sleep(0.3)
        
        sent_signal = state.get("sentiment_signal", {})
        yield {
            "type": "agent_done",
            "agent": "sentiment_analyst",
            "signal": sent_signal.get("signal", "HOLD"),
            "confidence": int(sent_signal.get("confidence", 0) * 100) if isinstance(sent_signal.get("confidence"), float) else sent_signal.get("confidence", 0),
            "reason": sent_signal.get("reasoning", ""),
            "summary": state.get("sentiment_analysis", "")[:150],
        }
    except Exception as e:
        yield {"type": "agent_done", "agent": "sentiment_analyst", "summary": f"Error: {e}"}
    
    # Check if agents have conflicting signals - run debate if needed
    tech_sig = state.get("technical_signal", {}).get("signal", "Neutral")
    fund_sig = state.get("fundamental_signal", {}).get("signal", "Neutral")
    sent_sig = state.get("sentiment_signal", {}).get("signal", "Neutral")
    
    signals = [tech_sig, fund_sig, sent_sig]
    has_conflict = len(set(signals)) > 1
    
    if has_conflict:
        yield {"type": "agent_start", "agent": "consensus_debate"}
        yield {
            "type": "debate_start",
            "message": "Agents have conflicting views. Starting team discussion...",
            "positions": {
                "Technical Analyst": tech_sig,
                "Fundamental Analyst": fund_sig,
                "Sentiment Analyst": sent_sig,
            },
        }
        await asyncio.sleep(0.3)
        
        # Run consensus debate with real LLM arguments
        from src.agents.consensus import AgentDebate
        from langchain_core.messages import HumanMessage
        
        debate = AgentDebate(client)
        llm = client.get_llm(temperature=0.7)
        
        # Get analysis summaries for context
        tech_analysis = str(state.get("technical_analysis", ""))
        fund_analysis = str(state.get("fundamental_analysis", ""))
        sent_analysis = str(state.get("sentiment_analysis", ""))
        
        # Get market research and date context
        market_research = state.get("market_research", {})
        research_timestamp = market_research.get("fetch_timestamp", "recently")
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        current_positions = {
            "Technical": tech_sig,
            "Fundamental": fund_sig,
            "Sentiment": sent_sig,
        }
        
        debate_history = []
        
        # Run debate rounds with real LLM arguments
        for round_num in range(1, 6):  # 5 rounds of debate
            # Check for early consensus before starting round
            if len(set(current_positions.values())) == 1 and round_num > 1:
                yield {
                    "type": "consensus_early_exit",
                    "message": f"Consensus reached early at round {round_num-1}!",
                    "decision": list(current_positions.values())[0]
                }
                break

            yield {
                "type": "debate_round",
                "round": round_num,
                "message": f"Round {round_num}: Agents presenting arguments...",
            }
            await asyncio.sleep(0.2)
            
            round_args = {}
            
            # Get relevant market context for each agent
            tech_market_data = market_research.get("technical_insights", "")[:300] if market_research.get("research_available") else ""
            fund_market_data = market_research.get("fundamental_research", "")[:300] if market_research.get("research_available") else ""
            sent_market_data = market_research.get("sentiment_sources", "")[:300] if market_research.get("research_available") else ""
            
            # Technical Analyst argues
            tech_prompt = f"""TODAY'S DATE: {current_date}

You are the Technical Analyst.
 
Argue for the {current_positions['Technical']} case on {ticker}.
Speak in first person ("I analyzed", "My conclusion", "Based on my research") and own the position.
Focus on IMPORTANT NUANCES and new insights. Do NOT repeat your entire analysis.
Keep your argument balanced (3-5 sentences).
Avoid self-labeling phrases like "as a Bullish/Neutral technical analyst". Focus on chart evidence.

Your analysis: {tech_analysis[:400]}

{f"Real-time technical insights (fetched {research_timestamp}): {tech_market_data}" if tech_market_data else ""}

IMPORTANT: Today is {current_date}. Reference CURRENT market conditions. Be persuasive and evidence-based."""
            tech_response = await asyncio.to_thread(llm.invoke, [HumanMessage(content=tech_prompt)])
            tech_arg = tech_response.content
            round_args["Technical"] = tech_arg
            
            yield {
                "type": "agent_argument",
                "agent": "Technical Analyst",
                "position": current_positions["Technical"],
                "message": tech_arg,
            }
            await asyncio.sleep(0.1)
            
            # Fundamental Analyst argues
            fund_prompt = f"""TODAY'S DATE: {current_date}

You are the Fundamental Analyst.
 
Argue for the {current_positions['Fundamental']} case on {ticker}.
Speak in first person ("I analyzed", "My conclusion", "Based on my research") and own the position.
Focus on IMPORTANT NUANCES and new insights. Do NOT repeat your entire analysis.
Keep your argument balanced (3-5 sentences). Do not ramble.
Avoid self-labeling phrases like "as a Neutral fundamental analyst". Focus on facts, valuation, and financials.

Your analysis: {fund_analysis[:400]}

{f"Real-time fundamental research (fetched {research_timestamp}): {fund_market_data}" if fund_market_data else ""}

IMPORTANT: Today is {current_date}. Reference CURRENT market conditions. Be persuasive and evidence-based."""
            fund_response = await asyncio.to_thread(llm.invoke, [HumanMessage(content=fund_prompt)])
            fund_arg = fund_response.content
            round_args["Fundamental"] = fund_arg
            
            yield {
                "type": "agent_argument",
                "agent": "Fundamental Analyst",
                "position": current_positions["Fundamental"],
                "message": fund_arg,
            }
            await asyncio.sleep(0.1)
            
            # Sentiment Analyst argues
            sent_prompt = f"""TODAY'S DATE: {current_date}

You are the Sentiment Analyst.
 
Argue for the {current_positions['Sentiment']} case on {ticker}.
Speak in first person ("I analyzed", "My conclusion", "Based on my research") and own the position.
Focus on IMPORTANT NUANCES and new insights. Do NOT repeat your entire analysis.
Keep your argument balanced (3-5 sentences).
Avoid self-labeling phrases like "as a Neutral sentiment analyst". Focus on evidence from news and sentiment.

Your analysis: {sent_analysis[:400]}

{f"Real-time sentiment sources (fetched {research_timestamp}): {sent_market_data}" if sent_market_data else ""}

IMPORTANT: Today is {current_date}. Reference CURRENT news and market sentiment. Be persuasive and evidence-based."""
            sent_response = await asyncio.to_thread(llm.invoke, [HumanMessage(content=sent_prompt)])
            sent_arg = sent_response.content
            round_args["Sentiment"] = sent_arg
            
            yield {
                "type": "agent_argument",
                "agent": "Sentiment Analyst",
                "position": current_positions["Sentiment"],
                "message": sent_arg,
            }
            await asyncio.sleep(0.1)
            
            debate_history.append(round_args)
            
            # After round 1, check if any agent wants to change position
            if round_num == 1:
                eval_prompt = f"""Based on this debate about {ticker}, should any analyst change their position?

Current positions: Technical={current_positions['Technical']}, Fundamental={current_positions['Fundamental']}, Sentiment={current_positions['Sentiment']}

Arguments:
- Technical: {tech_arg}
- Fundamental: {fund_arg}
- Sentiment: {sent_arg}

If an argument was very compelling, the other analysts might shift. Respond with updated positions:
Technical: [Bullish/Bearish/Neutral]
Fundamental: [Bullish/Bearish/Neutral]
Sentiment: [Bullish/Bearish/Neutral]"""
                
                eval_response = await asyncio.to_thread(llm.invoke, [HumanMessage(content=eval_prompt)])
                
                # Parse position changes
                for line in eval_response.content.split('\n'):
                    line = line.strip()
                    for analyst in ["Technical", "Fundamental", "Sentiment"]:
                        if line.startswith(f"{analyst}:"):
                            new_pos = line.split(":")[-1].strip()
                            if new_pos in ["Bullish", "Bearish", "Neutral"]:
                                if new_pos != current_positions[analyst]:
                                    yield {
                                        "type": "position_update",
                                        "round": round_num,
                                        "agent": f"{analyst} Analyst",
                                        "old_position": current_positions[analyst],
                                        "new_position": new_pos,
                                        "message": f"{analyst} Analyst changed from {current_positions[analyst]} to {new_pos}",
                                    }
                                    current_positions[analyst] = new_pos
        
        # Determine final consensus
        final_positions = list(current_positions.values())
        vote_counts = {s: final_positions.count(s) for s in set(final_positions)}
        consensus_decision = max(vote_counts, key=vote_counts.get)
        has_full_consensus = max(vote_counts.values()) == 3
        
        state["consensus_reached"] = has_full_consensus
        state["consensus_decision"] = consensus_decision
        state["debate_history"] = debate_history
        state["final_positions"] = current_positions
        
        if isinstance(state.get("technical_signal"), dict):
            state["technical_signal"] = {**state["technical_signal"], "signal": current_positions["Technical"]}
        else:
            state["technical_signal"] = {"signal": current_positions["Technical"]}
        
        if isinstance(state.get("fundamental_signal"), dict):
            state["fundamental_signal"] = {**state["fundamental_signal"], "signal": current_positions["Fundamental"]}
        else:
            state["fundamental_signal"] = {"signal": current_positions["Fundamental"]}
        
        if isinstance(state.get("sentiment_signal"), dict):
            state["sentiment_signal"] = {**state["sentiment_signal"], "signal": current_positions["Sentiment"]}
        else:
            state["sentiment_signal"] = {"signal": current_positions["Sentiment"]}
        
        yield {
            "type": "consensus_result",
            "consensus_reached": has_full_consensus,
            "decision": consensus_decision,
            "final_positions": current_positions,
            "message": f"Team {'reached consensus' if has_full_consensus else 'majority agrees'}: {consensus_decision}",
        }
        await asyncio.sleep(0.3)
        
        yield {"type": "agent_done", "agent": "consensus_debate", "consensus": consensus_decision}
    else:
        state["consensus_reached"] = True
        state["consensus_decision"] = signals[0]
        yield {
            "type": "consensus_result",
            "consensus_reached": True,
            "decision": signals[0],
            "message": f"All agents agree: {signals[0]}",
        }
    
    # Portfolio Manager
    yield {"type": "agent_start", "agent": "portfolio_manager"}
    yield {"type": "agent_thinking", "agent": "portfolio_manager", "content": "Presenting team's collective decision..."}
    await asyncio.sleep(0.5)
    
    try:
        state = portfolio_manager.analyze(state)
        
        manager_decision = state.get("manager_decision", {})
        decision = manager_decision.get("decision", "HOLD")
        confidence = manager_decision.get("confidence", 0)
        
        yield {
            "type": "agent_thinking",
            "agent": "portfolio_manager",
            "content": f"Final Decision: {decision} ({confidence}% confidence)"
        }
        await asyncio.sleep(0.3)
        
        yield {
            "type": "agent_done",
            "agent": "portfolio_manager",
            "signal": decision,
            "confidence": confidence,
            "reason": manager_decision.get("rationale", ""),
            "summary": state.get("final_thesis", "")[:150],
        }
        
    except Exception as e:
        yield {"type": "agent_done", "agent": "portfolio_manager", "summary": f"Error: {e}"}
    
    # Guardrails evaluation
    yield {"type": "agent_start", "agent": "guardrails"}
    yield {"type": "agent_thinking", "agent": "guardrails", "content": "Evaluating decision against guardrails..."}
    await asyncio.sleep(0.3)
    
    state = hitl_manager.evaluate_state(state)
    
    guardrail_results = state.get("guardrail_results", [])
    review_triggers = state.get("review_triggers", [])
    requires_review = state.get("requires_human_review", False)
    
    yield {
        "type": "agent_done",
        "agent": "guardrails",
        "requires_review": requires_review,
        "triggers": review_triggers,
        "results": guardrail_results,
    }
    
    # Human review if needed
    if requires_review:
        tech_signal = state.get("technical_signal", {})
        fund_signal = state.get("fundamental_signal", {})
        sent_signal = state.get("sentiment_signal", {})
        manager_decision = state.get("manager_decision", {})
        
        trigger_messages = {
            "low_confidence": "Low confidence in decision",
            "conflicting_signals": "Conflicting signals between analysts",
            "high_risk": "High-risk decision type",
            "large_position": "Large position size recommended",
            "volatile_market": "High market volatility detected",
        }
        
        trigger_text = ", ".join(trigger_messages.get(t, t) for t in review_triggers)
        
        yield {
            "type": "human_review",
            "question": f"Human review required for {ticker}: {trigger_text}",
            "context": f"Technical: {tech_signal.get('signal')}, Fundamental: {fund_signal.get('signal')}, Sentiment: {sent_signal.get('signal')}",
            "triggers": review_triggers,
            "review_id": state.get("pending_review_id"),
            # Include full state data for HITL runner
            "state": {
                "manager_decision": manager_decision,
                "technical_signal": tech_signal,
                "fundamental_signal": fund_signal,
                "sentiment_signal": sent_signal,
                "technical_analysis": state.get("technical_analysis", ""),
                "fundamental_analysis": state.get("fundamental_analysis", ""),
                "sentiment_analysis": state.get("sentiment_analysis", ""),
                "final_thesis": state.get("final_thesis", ""),
            },
        }
        await asyncio.sleep(0.5)
    
    # Final decision
    manager_decision = state.get("manager_decision", {})
    yield {
        "type": "final_decision",
        "decision": manager_decision.get("decision", "HOLD"),
        "confidence": manager_decision.get("confidence", 0),
        "requires_review": requires_review,
        "review_triggers": review_triggers,
    }
    
    yield {
        "type": "complete",
        "result": state,
    }
