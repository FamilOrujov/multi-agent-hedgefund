"""Tests for src/graph/guardrails.py module."""

import pytest


class TestGuardrailConfig:
    """Test GuardrailConfig dataclass."""

    def test_guardrail_config_defaults(self):
        """Test GuardrailConfig has correct defaults."""
        from src.graph.guardrails import GuardrailConfig

        config = GuardrailConfig()

        assert config.min_confidence_for_auto_approve == 0.75
        assert config.min_confidence_for_action == 0.50
        assert config.max_position_size_auto == "Moderate"
        assert config.volatility_threshold == 3.0
        assert config.require_consensus_for_auto is True
        assert config.min_agreeing_signals == 2
        assert config.min_data_points == 20
        assert config.max_data_age_days == 1
        assert "SELL" in config.always_review_decisions
        assert config.review_timeout_seconds == 300

    def test_guardrail_config_custom_values(self):
        """Test GuardrailConfig with custom values."""
        from src.graph.guardrails import GuardrailConfig

        config = GuardrailConfig(
            min_confidence_for_auto_approve=0.80,
            min_confidence_for_action=0.60,
            max_position_size_auto="Conservative",
        )

        assert config.min_confidence_for_auto_approve == 0.80
        assert config.min_confidence_for_action == 0.60
        assert config.max_position_size_auto == "Conservative"


class TestGuardrailResult:
    """Test GuardrailResult dataclass."""

    def test_guardrail_result_defaults(self):
        """Test GuardrailResult has correct defaults."""
        from src.graph.guardrails import GuardrailResult

        result = GuardrailResult(passed=True)

        assert result.passed is True
        assert result.trigger is None
        assert result.message == ""
        assert result.severity == "info"
        assert result.requires_human_review is False
        assert result.metadata == {}

    def test_guardrail_result_with_trigger(self):
        """Test GuardrailResult with trigger."""
        from src.graph.guardrails import GuardrailResult, ReviewTrigger

        result = GuardrailResult(
            passed=False,
            trigger=ReviewTrigger.LOW_CONFIDENCE,
            message="Confidence below threshold",
            severity="warning",
            requires_human_review=True,
        )

        assert result.passed is False
        assert result.trigger == ReviewTrigger.LOW_CONFIDENCE
        assert result.requires_human_review is True


class TestReviewTrigger:
    """Test ReviewTrigger enum."""

    def test_review_trigger_values(self):
        """Test ReviewTrigger has expected values."""
        from src.graph.guardrails import ReviewTrigger

        assert ReviewTrigger.LOW_CONFIDENCE.value == "low_confidence"
        assert ReviewTrigger.HIGH_RISK.value == "high_risk"
        assert ReviewTrigger.CONFLICTING_SIGNALS.value == "conflicting_signals"
        assert ReviewTrigger.LARGE_POSITION.value == "large_position"
        assert ReviewTrigger.VOLATILE_MARKET.value == "volatile_market"


class TestInputGuardrails:
    """Test InputGuardrails class."""

    def test_validate_ticker_valid(self):
        """Test ticker validation with valid tickers."""
        from src.graph.guardrails import InputGuardrails

        guardrails = InputGuardrails()

        result = guardrails.validate_ticker("AAPL")
        assert result.passed is True

        result = guardrails.validate_ticker("TSLA")
        assert result.passed is True

        result = guardrails.validate_ticker("MSFT")
        assert result.passed is True

    def test_validate_ticker_invalid_empty(self):
        """Test ticker validation with empty ticker."""
        from src.graph.guardrails import InputGuardrails, ReviewTrigger

        guardrails = InputGuardrails()

        result = guardrails.validate_ticker("")
        assert result.passed is False
        assert result.requires_human_review is False
        assert result.message == "Ticker symbol is required"

    def test_validate_ticker_invalid_too_long(self):
        """Test ticker validation with too long ticker."""
        from src.graph.guardrails import InputGuardrails, ReviewTrigger

        guardrails = InputGuardrails()

        result = guardrails.validate_ticker("TOOLONGTICKER")
        assert result.passed is False

    def test_validate_price_data_valid(self, sample_price_data):
        """Test price data validation with valid data."""
        from src.graph.guardrails import InputGuardrails

        guardrails = InputGuardrails()

        result = guardrails.validate_price_data(sample_price_data)
        # Should pass with valid data structure
        assert result is not None


class TestOutputGuardrails:
    """Test OutputGuardrails class."""

    def test_check_confidence_high(self):
        """Test confidence check with high confidence."""
        from src.graph.guardrails import OutputGuardrails

        guardrails = OutputGuardrails()

        result = guardrails.check_confidence(0.85)
        assert result.passed is True
        assert result.requires_human_review is False

    def test_check_confidence_low(self):
        """Test confidence check with low confidence."""
        from src.graph.guardrails import OutputGuardrails, ReviewTrigger

        guardrails = OutputGuardrails()

        result = guardrails.check_confidence(0.40)
        assert result.passed is False
        assert result.trigger == ReviewTrigger.LOW_CONFIDENCE
        assert result.requires_human_review is True

    def test_check_confidence_medium(self):
        """Test confidence check with medium confidence."""
        from src.graph.guardrails import OutputGuardrails

        guardrails = OutputGuardrails()

        result = guardrails.check_confidence(0.60)
        # Medium confidence should require review but not fail
        assert result.requires_human_review is True

    def test_check_decision_type_sell(self):
        """Test decision type check for SELL."""
        from src.graph.guardrails import OutputGuardrails, ReviewTrigger

        guardrails = OutputGuardrails()

        result = guardrails.check_decision_type("SELL")
        assert result.requires_human_review is True
        assert result.trigger == ReviewTrigger.HIGH_RISK

    def test_check_decision_type_buy(self):
        """Test decision type check for BUY."""
        from src.graph.guardrails import OutputGuardrails

        guardrails = OutputGuardrails()

        result = guardrails.check_decision_type("BUY")
        assert result.passed is True
        assert result.requires_human_review is False

    def test_check_signal_consensus_agreement(self):
        """Test signal consensus with agreeing signals."""
        from src.graph.guardrails import OutputGuardrails

        guardrails = OutputGuardrails()

        signals = {
            "technical": {"signal": "Bullish"},
            "fundamental": {"signal": "Bullish"},
            "sentiment": {"signal": "Bullish"},
        }

        result = guardrails.check_signal_consensus(signals)
        assert result.passed is True

    def test_check_signal_consensus_disagreement(self):
        """Test signal consensus with conflicting signals."""
        from src.graph.guardrails import OutputGuardrails, ReviewTrigger

        guardrails = OutputGuardrails()

        signals = {
            "technical": {"signal": "Bullish"},
            "fundamental": {"signal": "Bearish"},
            "sentiment": {"signal": "Neutral"},
        }

        result = guardrails.check_signal_consensus(signals)
        # Should pass but trigger review
        assert result.passed is True
        assert result.requires_human_review is True
        assert result.trigger == ReviewTrigger.CONFLICTING_SIGNALS
        assert result.trigger == ReviewTrigger.CONFLICTING_SIGNALS


class TestHITLManager:
    """Test HITLManager class."""

    def test_hitl_manager_initialization(self):
        """Test HITLManager initializes correctly."""
        from src.graph.guardrails import HITLManager

        manager = HITLManager()

        assert manager.config is not None
        assert manager.input_guardrails is not None
        assert manager.output_guardrails is not None

    def test_hitl_manager_custom_config(self):
        """Test HITLManager with custom config."""
        from src.graph.guardrails import HITLManager, GuardrailConfig

        config = GuardrailConfig(min_confidence_for_auto_approve=0.90)
        manager = HITLManager(config=config)

        assert manager.config.min_confidence_for_auto_approve == 0.90

    def test_hitl_manager_get_pending_reviews(self):
        """Test getting pending reviews."""
        from src.graph.guardrails import HITLManager

        manager = HITLManager()
        pending = manager.get_pending_reviews()

        assert isinstance(pending, list)
