
from typing import Any, Optional, AsyncGenerator
import asyncio

from src.llm.ollama_client import OllamaClient
from langchain_core.messages import HumanMessage, SystemMessage


class AgentDebate:
    """Facilitates debate between agents to reach consensus."""
    
    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client
        self.llm = ollama_client.get_llm(temperature=0.6)
        self.debate_history = []
        self.max_rounds = 5
    
    async def run_debate(
        self,
        state: dict[str, Any],
        stream_callback: Optional[callable] = None,
    ) -> dict[str, Any]:
        """
        Run a multi-round debate between agents to reach consensus.
        
        Returns updated state with consensus decision.
        """
        ticker = state.get("ticker", "UNKNOWN")
        
        # Get initial positions
        tech_signal = state.get("technical_signal", {})
        fund_signal = state.get("fundamental_signal", {})
        sent_signal = state.get("sentiment_signal", {})
        
        tech_pos = tech_signal.get("signal", "Neutral")
        fund_pos = fund_signal.get("signal", "Neutral")
        sent_pos = sent_signal.get("signal", "Neutral")
        
        # Check if already in consensus
        positions = [tech_pos, fund_pos, sent_pos]
        if len(set(positions)) == 1:
            if stream_callback:
                await stream_callback({
                    "type": "consensus_reached",
                    "message": f"All agents agree: {positions[0]}",
                    "rounds": 0,
                })
            return {
                **state,
                "consensus_reached": True,
                "consensus_decision": positions[0],
                "debate_summary": f"Unanimous agreement: {positions[0]}",
            }
        
        # Start debate
        self.debate_history = []
        
        if stream_callback:
            await stream_callback({
                "type": "debate_start",
                "message": f"Agents have conflicting views. Starting debate...",
                "positions": {
                    "Technical Analyst": tech_pos,
                    "Fundamental Analyst": fund_pos,
                    "Sentiment Analyst": sent_pos,
                },
            })
        
        # Run debate rounds
        for round_num in range(1, self.max_rounds + 1):
            if stream_callback:
                await stream_callback({
                    "type": "debate_round",
                    "round": round_num,
                    "message": f"Round {round_num} of debate",
                })
            
            # Each agent argues their position
            round_arguments = {}
            market_research = state.get("market_research", {})
            
            # Technical Analyst argues
            tech_arg = await self._agent_argue(
                "Technical Analyst",
                tech_pos,
                state.get("technical_analysis", ""),
                self.debate_history,
                ticker,
                market_research,
            )
            round_arguments["Technical Analyst"] = tech_arg
            if stream_callback:
                await stream_callback({
                    "type": "agent_argument",
                    "agent": "Technical Analyst",
                    "position": tech_pos,
                    "argument": tech_arg,
                })
            
            # Fundamental Analyst argues
            fund_arg = await self._agent_argue(
                "Fundamental Analyst",
                fund_pos,
                state.get("fundamental_analysis", ""),
                self.debate_history,
                ticker,
                market_research,
            )
            round_arguments["Fundamental Analyst"] = fund_arg
            if stream_callback:
                await stream_callback({
                    "type": "agent_argument",
                    "agent": "Fundamental Analyst",
                    "position": fund_pos,
                    "argument": fund_arg,
                })
            
            # Sentiment Analyst argues
            sent_arg = await self._agent_argue(
                "Sentiment Analyst",
                sent_pos,
                state.get("sentiment_analysis", ""),
                self.debate_history,
                ticker,
                market_research,
            )
            round_arguments["Sentiment Analyst"] = sent_arg
            if stream_callback:
                await stream_callback({
                    "type": "agent_argument",
                    "agent": "Sentiment Analyst",
                    "position": sent_pos,
                    "argument": sent_arg,
                })
            
            self.debate_history.append({
                "round": round_num,
                "arguments": round_arguments,
            })
            
            # Check if any agent wants to change position
            new_positions = await self._evaluate_positions(
                ticker,
                round_arguments,
                {"Technical": tech_pos, "Fundamental": fund_pos, "Sentiment": sent_pos},
            )
            
            tech_pos = new_positions.get("Technical", tech_pos)
            fund_pos = new_positions.get("Fundamental", fund_pos)
            sent_pos = new_positions.get("Sentiment", sent_pos)
            
            if stream_callback:
                await stream_callback({
                    "type": "position_update",
                    "round": round_num,
                    "positions": {
                        "Technical Analyst": tech_pos,
                        "Fundamental Analyst": fund_pos,
                        "Sentiment Analyst": sent_pos,
                    },
                })
            
            # Check for consensus
            positions = [tech_pos, fund_pos, sent_pos]
            if len(set(positions)) == 1:
                if stream_callback:
                    await stream_callback({
                        "type": "consensus_reached",
                        "message": f"Consensus reached after {round_num} rounds: {positions[0]}",
                        "rounds": round_num,
                    })
                return {
                    **state,
                    "consensus_reached": True,
                    "consensus_decision": positions[0],
                    "debate_history": self.debate_history,
                    "debate_summary": self._generate_debate_summary(round_num, positions[0]),
                }
        
        # No consensus after max rounds - use majority vote
        final_decision = self._majority_vote(tech_pos, fund_pos, sent_pos)
        
        if stream_callback:
            await stream_callback({
                "type": "majority_vote",
                "message": f"No full consensus. Majority decision: {final_decision}",
                "final_positions": {
                    "Technical Analyst": tech_pos,
                    "Fundamental Analyst": fund_pos,
                    "Sentiment Analyst": sent_pos,
                },
            })
        
        return {
            **state,
            "consensus_reached": False,
            "consensus_decision": final_decision,
            "majority_vote": True,
            "debate_history": self.debate_history,
            "debate_summary": self._generate_debate_summary(self.max_rounds, final_decision, majority=True),
            "final_positions": {
                "Technical": tech_pos,
                "Fundamental": fund_pos,
                "Sentiment": sent_pos,
            },
        }
    
    async def _agent_argue(
        self,
        agent_name: str,
        position: str,
        analysis: str,
        history: list,
        ticker: str,
        market_research: dict = None,
    ) -> str:
        """Have an agent argue for their position."""
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        history_text = ""
        if history:
            for h in history[-2:]:  # Last 2 rounds
                for agent, arg in h.get("arguments", {}).items():
                    history_text += f"\n{agent}: {arg[:200]}..."
        
        # Add relevant market context based on agent type
        market_context = ""
        research_timestamp = ""
        if market_research and market_research.get("research_available"):
            research_timestamp = market_research.get("fetch_timestamp", "recently")
            if "Technical" in agent_name:
                market_context = market_research.get("technical_insights", "")[:400]
            elif "Fundamental" in agent_name:
                market_context = market_research.get("fundamental_research", "")[:400]
            elif "Sentiment" in agent_name:
                market_context = market_research.get("sentiment_sources", "")[:400]
        
        research_section = ""
        if market_context:
            research_section = f"\n\n⚠️ REAL-TIME market data (fetched: {research_timestamp}) - CITE THIS in your argument:\n{market_context}"
        
        prompt = f"""TODAY'S DATE: {current_date}

You are the {agent_name} in a hedge fund team debating about {ticker}.

Assigned signal to argue: {position}

Your analysis summary:
{analysis[:500]}
{research_section}

Previous debate points:
{history_text if history_text else "This is the first round."}

IMPORTANT:
- Today is {current_date}. Reference CURRENT market conditions, not outdated information.
- If real-time market data is provided above, CITE IT in your argument.
- Argue for the {position} case in 3-5 sentences. Keep it balanced.
- Speak in first person ("I analyzed", "My conclusion", "Based on my research") and own the position.
- Do not self-label (e.g., "as a Neutral fundamental analyst") and do not present the signal as an identity.
- If you see strong counter-arguments, you may acknowledge them.
- Focus on the strongest evidence supporting this view."""

        messages = [HumanMessage(content=prompt)]
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content
    
    async def _evaluate_positions(
        self,
        ticker: str,
        arguments: dict[str, str],
        current_positions: dict[str, str],
    ) -> dict[str, str]:
        """Evaluate if any agent should change their position based on debate."""
        
        prompt = f"""Based on the following debate about {ticker}, determine if any analyst should change their position.

Current positions:
- Technical Analyst: {current_positions['Technical']}
- Fundamental Analyst: {current_positions['Fundamental']}
- Sentiment Analyst: {current_positions['Sentiment']}

Arguments made:
- Technical: {arguments.get('Technical Analyst', '')[:300]}
- Fundamental: {arguments.get('Fundamental Analyst', '')[:300]}
- Sentiment: {arguments.get('Sentiment Analyst', '')[:300]}

If an analyst's argument was weak or another made a compelling counter-point, they might change.
Respond with the updated positions in this exact format:
Technical: [Bullish/Bearish/Neutral]
Fundamental: [Bullish/Bearish/Neutral]
Sentiment: [Bullish/Bearish/Neutral]

Only change a position if there's a strong reason. Analysts tend to hold their views unless convinced."""

        messages = [HumanMessage(content=prompt)]
        response = await asyncio.to_thread(self.llm.invoke, messages)
        
        # Parse response
        new_positions = current_positions.copy()
        for line in response.content.split('\n'):
            line = line.strip()
            if line.startswith("Technical:"):
                pos = line.split(":")[-1].strip()
                if pos in ["Bullish", "Bearish", "Neutral"]:
                    new_positions["Technical"] = pos
            elif line.startswith("Fundamental:"):
                pos = line.split(":")[-1].strip()
                if pos in ["Bullish", "Bearish", "Neutral"]:
                    new_positions["Fundamental"] = pos
            elif line.startswith("Sentiment:"):
                pos = line.split(":")[-1].strip()
                if pos in ["Bullish", "Bearish", "Neutral"]:
                    new_positions["Sentiment"] = pos
        
        return new_positions
    
    def _majority_vote(self, tech: str, fund: str, sent: str) -> str:
        """Determine decision by majority vote."""
        votes = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
        for pos in [tech, fund, sent]:
            if pos in votes:
                votes[pos] += 1
        
        max_votes = max(votes.values())
        for decision, count in votes.items():
            if count == max_votes:
                return decision
        return "Neutral"
    
    def _generate_debate_summary(
        self,
        rounds: int,
        decision: str,
        majority: bool = False,
    ) -> str:
        """Generate a summary of the debate."""
        if majority:
            return f"After {rounds} rounds of debate, the team reached a majority decision of {decision}. While not all analysts agreed, the prevailing view was supported by the strongest evidence."
        else:
            return f"After {rounds} rounds of constructive debate, all analysts reached consensus on {decision}. The team unified their analysis through discussion and evidence sharing."
