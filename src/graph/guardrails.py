
from typing import Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph


class ReviewTrigger(Enum):
    """Triggers that require human review."""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_RISK = "high_risk"
    CONFLICTING_SIGNALS = "conflicting_signals"
    LARGE_POSITION = "large_position"
    VOLATILE_MARKET = "volatile_market"
    UNUSUAL_PATTERN = "unusual_pattern"
    DATA_QUALITY = "data_quality"


@dataclass
class GuardrailConfig:
    """Configuration for guardrails."""
    # Confidence thresholds
    min_confidence_for_auto_approve: float = 0.75
    min_confidence_for_action: float = 0.50
    
    # Risk thresholds
    max_position_size_auto: str = "Moderate"  # Conservative, Moderate, Aggressive
    volatility_threshold: float = 3.0  # ATR percentage
    
    # Signal agreement
    require_consensus_for_auto: bool = True
    min_agreeing_signals: int = 2
    
    # Data quality
    min_data_points: int = 20
    max_data_age_days: int = 1
    
    # Human review settings
    always_review_decisions: list[str] = field(default_factory=lambda: ["SELL", "STRONG_SELL"])
    review_timeout_seconds: int = 300


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    trigger: Optional[ReviewTrigger] = None
    message: str = ""
    severity: str = "info"  # info, warning, critical
    requires_human_review: bool = False
    metadata: dict = field(default_factory=dict)


class InputGuardrails:
    """Validates inputs before processing."""
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
    
    def validate_ticker(self, ticker: str) -> GuardrailResult:
        """Validate ticker symbol format."""
        if not ticker:
            return GuardrailResult(
                passed=False,
                message="Ticker symbol is required",
                severity="critical"
            )
        
        ticker = ticker.upper().strip()
        
        # Basic validation
        if not ticker.isalpha() or len(ticker) > 5:
            return GuardrailResult(
                passed=False,
                message=f"Invalid ticker format: {ticker}",
                severity="critical"
            )
        
        return GuardrailResult(passed=True, message="Ticker validated")
    
    def validate_price_data(self, price_data: dict) -> GuardrailResult:
        """Validate price data quality."""
        if not price_data:
            return GuardrailResult(
                passed=False,
                trigger=ReviewTrigger.DATA_QUALITY,
                message="No price data available",
                severity="critical",
                requires_human_review=True
            )
        
        # Check data points
        data_points = len(price_data.get("Close", []))
        if data_points < self.config.min_data_points:
            return GuardrailResult(
                passed=False,
                trigger=ReviewTrigger.DATA_QUALITY,
                message=f"Insufficient data points: {data_points} < {self.config.min_data_points}",
                severity="warning",
                requires_human_review=True
            )
        
        return GuardrailResult(passed=True, message="Price data validated")


class OutputGuardrails:
    """Validates outputs and determines if human review is needed."""
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
    
    def check_confidence(self, confidence: float) -> GuardrailResult:
        """Check if confidence meets thresholds."""
        if confidence < self.config.min_confidence_for_action:
            return GuardrailResult(
                passed=False,
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                message=f"Confidence too low: {confidence:.0%} < {self.config.min_confidence_for_action:.0%}",
                severity="warning",
                requires_human_review=True,
                metadata={"confidence": confidence}
            )
        
        if confidence < self.config.min_confidence_for_auto_approve:
            return GuardrailResult(
                passed=True,
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                message=f"Confidence below auto-approve threshold: {confidence:.0%}",
                severity="info",
                requires_human_review=True,
                metadata={"confidence": confidence}
            )
        
        return GuardrailResult(passed=True, message="Confidence acceptable")
    
    def check_signal_consensus(self, signals: dict[str, dict]) -> GuardrailResult:
        """Check if signals are in agreement."""
        signal_values = []
        for name, signal_data in signals.items():
            if isinstance(signal_data, dict) and "signal" in signal_data:
                signal_values.append(signal_data["signal"])
        
        if not signal_values:
            return GuardrailResult(
                passed=False,
                trigger=ReviewTrigger.DATA_QUALITY,
                message="No signals available",
                severity="critical",
                requires_human_review=True
            )
        
        # Check for conflicts
        bullish = sum(1 for s in signal_values if "Bullish" in s or s in ["BUY", "STRONG_BUY"])
        bearish = sum(1 for s in signal_values if "Bearish" in s or s in ["SELL", "STRONG_SELL"])
        
        if bullish > 0 and bearish > 0:
            return GuardrailResult(
                passed=True,
                trigger=ReviewTrigger.CONFLICTING_SIGNALS,
                message=f"Conflicting signals: {bullish} bullish, {bearish} bearish",
                severity="warning",
                requires_human_review=True,
                metadata={"bullish": bullish, "bearish": bearish, "signals": signal_values}
            )
        
        # Check consensus requirement
        if self.config.require_consensus_for_auto:
            max_agreement = max(bullish, bearish, len(signal_values) - bullish - bearish)
            if max_agreement < self.config.min_agreeing_signals:
                return GuardrailResult(
                    passed=True,
                    trigger=ReviewTrigger.CONFLICTING_SIGNALS,
                    message="Insufficient signal agreement for auto-approval",
                    severity="info",
                    requires_human_review=True
                )
        
        return GuardrailResult(passed=True, message="Signals in consensus")
    
    def check_position_size(self, position_size: str, decision: str) -> GuardrailResult:
        """Check if position size is within auto-approve limits."""
        size_order = {"None": 0, "Conservative": 1, "Moderate": 2, "Aggressive": 3}
        
        current = size_order.get(position_size, 0)
        max_auto = size_order.get(self.config.max_position_size_auto, 2)
        
        if current > max_auto:
            return GuardrailResult(
                passed=True,
                trigger=ReviewTrigger.LARGE_POSITION,
                message=f"Position size '{position_size}' exceeds auto-approve limit",
                severity="warning",
                requires_human_review=True,
                metadata={"position_size": position_size, "decision": decision}
            )
        
        return GuardrailResult(passed=True, message="Position size acceptable")
    
    def check_decision_type(self, decision: str) -> GuardrailResult:
        """Check if decision type requires review."""
        if decision in self.config.always_review_decisions:
            return GuardrailResult(
                passed=True,
                trigger=ReviewTrigger.HIGH_RISK,
                message=f"Decision '{decision}' always requires human review",
                severity="info",
                requires_human_review=True,
                metadata={"decision": decision}
            )
        
        return GuardrailResult(passed=True, message="Decision type acceptable")
    
    def check_volatility(self, indicators: dict) -> GuardrailResult:
        """Check market volatility levels."""
        atr = indicators.get("atr", {})
        volatility = atr.get("volatility", "")
        
        if volatility == "High volatility":
            return GuardrailResult(
                passed=True,
                trigger=ReviewTrigger.VOLATILE_MARKET,
                message="High market volatility detected",
                severity="warning",
                requires_human_review=True,
                metadata={"volatility": volatility, "atr": atr.get("value")}
            )
        
        return GuardrailResult(passed=True, message="Volatility acceptable")


class HITLManager:
    """Manages Human-in-the-Loop interactions."""
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self.input_guardrails = InputGuardrails(config)
        self.output_guardrails = OutputGuardrails(config)
        self.pending_reviews: dict[str, dict] = {}
        self.review_history: list[dict] = []
    
    def evaluate_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate state and determine if human review is needed.
        
        Returns updated state with review requirements.
        """
        results = []
        requires_review = False
        triggers = []
        
        # Check confidence
        manager_decision = state.get("manager_decision", {})
        confidence = manager_decision.get("confidence", 0) / 100  # Convert to 0-1
        conf_result = self.output_guardrails.check_confidence(confidence)
        results.append(conf_result)
        if conf_result.requires_human_review:
            requires_review = True
            if conf_result.trigger:
                triggers.append(conf_result.trigger.value)
        
        # Check signal consensus
        signals = {
            "technical": state.get("technical_signal", {}),
            "fundamental": state.get("fundamental_signal", {}),
            "sentiment": state.get("sentiment_signal", {}),
        }
        consensus_result = self.output_guardrails.check_signal_consensus(signals)
        results.append(consensus_result)
        if consensus_result.requires_human_review:
            requires_review = True
            if consensus_result.trigger:
                triggers.append(consensus_result.trigger.value)
        
        # Check position size
        decision = manager_decision.get("decision", "HOLD")
        position_size = manager_decision.get("position_size", "None")
        pos_result = self.output_guardrails.check_position_size(position_size, decision)
        results.append(pos_result)
        if pos_result.requires_human_review:
            requires_review = True
            if pos_result.trigger:
                triggers.append(pos_result.trigger.value)
        
        # Check decision type
        dec_result = self.output_guardrails.check_decision_type(decision)
        results.append(dec_result)
        if dec_result.requires_human_review:
            requires_review = True
            if dec_result.trigger:
                triggers.append(dec_result.trigger.value)
        
        # Check volatility
        indicators = state.get("technical_indicators", {})
        vol_result = self.output_guardrails.check_volatility(indicators)
        results.append(vol_result)
        if vol_result.requires_human_review:
            requires_review = True
            if vol_result.trigger:
                triggers.append(vol_result.trigger.value)
        
        # Update state
        state["requires_human_review"] = requires_review
        state["review_triggers"] = list(set(triggers))
        state["guardrail_results"] = [
            {
                "passed": r.passed,
                "message": r.message,
                "severity": r.severity,
                "trigger": r.trigger.value if r.trigger else None,
            }
            for r in results
        ]
        
        return state
    
    def create_review_request(
        self,
        state: dict[str, Any],
        review_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a human review request."""
        review_id = review_id or f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        ticker = state.get("ticker", "UNKNOWN")
        decision = state.get("manager_decision", {})
        
        review_request = {
            "review_id": review_id,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "decision": decision.get("decision", "HOLD"),
            "confidence": decision.get("confidence", 0),
            "position_size": decision.get("position_size", "None"),
            "triggers": state.get("review_triggers", []),
            "signals": {
                "technical": state.get("technical_signal", {}),
                "fundamental": state.get("fundamental_signal", {}),
                "sentiment": state.get("sentiment_signal", {}),
            },
            "thesis_summary": state.get("final_thesis", "")[:500],
            "status": "pending",
            "human_decision": None,
            "human_notes": None,
        }
        
        self.pending_reviews[review_id] = review_request
        return review_request
    
    def submit_review(
        self,
        review_id: str,
        approved: bool,
        modified_decision: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Submit human review decision."""
        if review_id not in self.pending_reviews:
            raise ValueError(f"Review {review_id} not found")
        
        review = self.pending_reviews.pop(review_id)
        review["status"] = "approved" if approved else "rejected"
        review["human_decision"] = modified_decision or review["decision"]
        review["human_notes"] = notes
        review["reviewed_at"] = datetime.now().isoformat()
        
        self.review_history.append(review)
        
        return review
    
    def get_pending_reviews(self) -> list[dict]:
        """Get all pending reviews."""
        return list(self.pending_reviews.values())


def create_guardrail_node(hitl_manager: HITLManager):
    """Create a guardrail node for LangGraph."""
    
    def guardrail_node(state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate state and apply guardrails."""
        return hitl_manager.evaluate_state(state)
    
    return guardrail_node


def create_human_review_node(hitl_manager: HITLManager):
    """Create a human review node for LangGraph."""
    
    def human_review_node(state: dict[str, Any]) -> dict[str, Any]:
        """Prepare state for human review (interrupt point)."""
        if state.get("requires_human_review"):
            review_request = hitl_manager.create_review_request(state)
            state["pending_review_id"] = review_request["review_id"]
            state["status"] = "awaiting_review"
        return state
    
    return human_review_node
