
from typing import Any, Optional

from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.data.tavily_tools import search_stock_news, search_risk_factors


class PortfolioManagerAgent(BaseAgent):
    """Agent responsible for synthesizing analyses and making final investment decisions."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.4,
    ):
        super().__init__(ollama_client, model, temperature)

    @property
    def name(self) -> str:
        return "portfolio_manager"

    @property
    def role(self) -> str:
        return "Portfolio Manager - Chief Investment Decision Maker"

    @property
    def system_prompt(self) -> str:
        return """You are the Portfolio Manager facilitating a hedge fund analysis team discussion.

You speak on behalf of the TEAM, not as an individual. Always use "We" not "I".

Your responsibilities:
1. Synthesize reports from Technical, Fundamental, and Sentiment analysts
2. Facilitate discussion to resolve disagreements
3. Present the TEAM's collective investment decision
4. Provide clear reasoning based on team consensus

Decision Framework:
- Consider all three analyst signals (Technical, Fundamental, Sentiment)
- Weight based on signal confidence and team discussion
- Explain how the team resolved any conflicting views
- Account for risk factors and position sizing

Your output must include:
- Final Decision: BUY / SELL / HOLD (as a TEAM decision)
- Confidence Score: 0-100%
- Position Size Recommendation: Conservative / Moderate / Aggressive
- Key Reasoning: Why WE (the team) made this decision
- Risk Factors: What could go wrong
- How consensus was reached or dissenting views

IMPORTANT: Always say "We have decided" or "The team recommends" - NEVER "I have decided".
This is a collaborative multi-agent decision, not an individual one.
"""

    def _aggregate_signals(self, state: dict[str, Any]) -> dict[str, Any]:
        """Aggregate signals from all analysts."""
        signals = {
            "technical": state.get("technical_signal", {}),
            "fundamental": state.get("fundamental_signal", {}),
            "sentiment": state.get("sentiment_signal", {}),
        }

        signal_values = {"Bullish": 1, "Neutral": 0, "Bearish": -1}

        weighted_sum = 0
        total_weight = 0
        conflicts = []

        for analyst, signal_data in signals.items():
            if isinstance(signal_data, dict) and "signal" in signal_data:
                signal = signal_data.get("signal", "Neutral")
                confidence = signal_data.get("confidence", 0.5)

                value = signal_values.get(signal, 0)
                weighted_sum += value * confidence
                total_weight += confidence

        if total_weight > 0:
            consensus_score = weighted_sum / total_weight
        else:
            consensus_score = 0

        signal_list = []
        for analyst, signal_data in signals.items():
            if isinstance(signal_data, dict) and "signal" in signal_data:
                signal_list.append(signal_data.get("signal", "Neutral"))

        if len(set(signal_list)) > 1:
            conflicts = [
                f"{analyst}: {signals[analyst].get('signal', 'N/A')}"
                for analyst in signals
                if isinstance(signals[analyst], dict)
            ]

        return {
            "consensus_score": round(consensus_score, 2),
            "individual_signals": signals,
            "conflicts": conflicts,
            "has_consensus": len(set(signal_list)) == 1,
        }

    def _determine_decision(self, consensus_score: float) -> tuple[str, int]:
        """Determine final decision based on consensus score."""
        if consensus_score >= 0.5:
            return "BUY", int(60 + consensus_score * 40)
        elif consensus_score <= -0.5:
            return "SELL", int(60 + abs(consensus_score) * 40)
        elif consensus_score >= 0.2:
            return "BUY", int(50 + consensus_score * 30)
        elif consensus_score <= -0.2:
            return "SELL", int(50 + abs(consensus_score) * 30)
        else:
            return "HOLD", int(50 + (1 - abs(consensus_score)) * 20)

    def _determine_position_size(
        self,
        decision: str,
        confidence: int,
        has_consensus: bool,
    ) -> str:
        """Determine recommended position size."""
        if decision == "HOLD":
            return "None"

        if confidence >= 80 and has_consensus:
            return "Aggressive"
        elif confidence >= 60:
            return "Moderate"
        else:
            return "Conservative"

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        """Synthesize all analyses and make final decision."""
        ticker = state.get("ticker", "")

        aggregation = self._aggregate_signals(state)
        
        # Use consensus decision from debate if available
        consensus_decision = state.get("consensus_decision")
        if consensus_decision:
            # Map consensus signal to action
            if consensus_decision == "Bullish":
                decision = "BUY"
                confidence = 75 if state.get("consensus_reached") else 65
            elif consensus_decision == "Bearish":
                decision = "SELL"
                confidence = 75 if state.get("consensus_reached") else 65
            else:
                decision = "HOLD"
                confidence = 70 if state.get("consensus_reached") else 60
        else:
            decision, confidence = self._determine_decision(aggregation["consensus_score"])
        position_size = self._determine_position_size(
            decision,
            confidence,
            aggregation["has_consensus"],
        )

        technical_analysis = state.get("technical_analysis", "Not available")
        fundamental_analysis = state.get("fundamental_analysis", "Not available")
        sentiment_analysis = state.get("sentiment_analysis", "Not available")

        # Get risk intelligence from market research
        market_research = state.get("market_research", {})
        risk_factors = ""
        if market_research.get("research_available"):
            risk_factors = market_research.get("risk_factors", "")[:1500]
        
        # Risk context for decision
        risk_context = ""
        if risk_factors:
            risk_context = f"""## Latest Risk Intelligence:
{risk_factors}
"""

        consensus_status = "unanimous agreement" if aggregation['has_consensus'] else "majority decision after team discussion"
        
        # Get current date for context
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        research_timestamp = market_research.get("fetch_timestamp", "recently")
        
        synthesis_prompt = f"""TODAY'S DATE: {current_date}

As the Portfolio Manager, present the TEAM's collective decision for {ticker}.

Remember: Use "We" and "The team" - NEVER "I". This is a collaborative decision.

## Technical Analyst's View:
{technical_analysis}

## Fundamental Analyst's View:
{fundamental_analysis}

## Sentiment Analyst's View:
{sentiment_analysis}

## Team Discussion Summary:
- Consensus Score: {aggregation['consensus_score']} (-1 = Strong Sell, +1 = Strong Buy)
- Agreement Status: {consensus_status}
- Differing Views: {aggregation['conflicts'] if aggregation['conflicts'] else 'None - all analysts aligned'}

## ⚠️ Real-Time Risk Intelligence (Fetched: {research_timestamp}):
{risk_factors if risk_factors else 'No real-time risk data available'}

## Team's Collective Decision:
- Decision: {decision}
- Confidence: {confidence}%
- Position Size: {position_size}

⚠️ CRITICAL: Today is {current_date}. Your synthesis should reflect the CURRENT market conditions.
Use the real-time risk intelligence above to inform your risk assessment.

Output the final thesis in clear Markdown with these exact section headings:
## Team Decision
## Rationale
## Risks
## Position Size & Time Horizon
## Consensus Notes

Under each heading, write 2-4 sentences or short bullet points.
Start the Team Decision section with a sentence like "We have decided...".

IMPORTANT: Write as if presenting a team decision. Use phrases like:
- "We have decided..."
- "The team recommends..."
- "Our analysts concluded..."
- "After team discussion, we..."
"""

        final_thesis = self.invoke(synthesis_prompt)

        iteration_count = state.get("iteration_count", 0) + 1

        # Use consensus from debate if available
        consensus_reached = state.get("consensus_reached", aggregation["has_consensus"])
        
        return {
            **state,
            "signal_aggregation": aggregation,
            "manager_decision": {
                "decision": decision,
                "confidence": confidence,
                "position_size": position_size,
                "has_consensus": consensus_reached,
            },
            "final_thesis": final_thesis,
            "iteration_count": iteration_count,
        }

    def should_approve(self, state: dict[str, Any]) -> bool:
        """Determine if the analysis is complete and ready for human review."""
        decision = state.get("manager_decision", {})
        confidence = decision.get("confidence", 0)
        iteration_count = state.get("iteration_count", 0)

        if confidence >= 70:
            return True

        if iteration_count >= 3:
            return True

        return False

    def get_critique_feedback(self, state: dict[str, Any]) -> str:
        """Generate feedback for another iteration if needed."""
        decision = state.get("manager_decision", {})
        aggregation = state.get("signal_aggregation", {})

        if not aggregation.get("has_consensus"):
            return (
                "The analysts have conflicting views. Please re-examine the data "
                "and provide more definitive signals with supporting evidence."
            )

        if decision.get("confidence", 0) < 60:
            return (
                "The confidence level is too low. Please gather additional data "
                "or provide stronger justification for your signals."
            )

        return "Analysis needs refinement. Please review and strengthen your conclusions."
