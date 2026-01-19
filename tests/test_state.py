
from datetime import datetime

import pytest


class TestSignalData:
    """Test SignalData model."""

    def test_signal_data_defaults(self):
        """Test SignalData has correct defaults."""
        from src.state.schema import SignalData

        signal = SignalData()

        assert signal.signal == "Neutral"
        assert signal.confidence == 0.0
        assert signal.score is None

    def test_signal_data_custom_values(self):
        """Test SignalData with custom values."""
        from src.state.schema import SignalData

        signal = SignalData(signal="Bullish", confidence=0.85, score=7.5)

        assert signal.signal == "Bullish"
        assert signal.confidence == 0.85
        assert signal.score == 7.5

    def test_signal_data_confidence_bounds(self):
        """Test SignalData confidence is bounded 0-1."""
        from src.state.schema import SignalData

        # Valid bounds
        signal_low = SignalData(confidence=0.0)
        signal_high = SignalData(confidence=1.0)

        assert signal_low.confidence == 0.0
        assert signal_high.confidence == 1.0


class TestManagerDecision:
    """Test ManagerDecision model."""

    def test_manager_decision_defaults(self):
        """Test ManagerDecision has correct defaults."""
        from src.state.schema import ManagerDecision

        decision = ManagerDecision()

        assert decision.decision == "HOLD"
        assert decision.confidence == 0
        assert decision.position_size == "None"
        assert decision.has_consensus is False

    def test_manager_decision_custom_values(self):
        """Test ManagerDecision with custom values."""
        from src.state.schema import ManagerDecision

        decision = ManagerDecision(
            decision="BUY",
            confidence=85,
            position_size="Aggressive",
            has_consensus=True,
        )

        assert decision.decision == "BUY"
        assert decision.confidence == 85
        assert decision.position_size == "Aggressive"
        assert decision.has_consensus is True


class TestAnalysisResult:
    """Test AnalysisResult model."""

    def test_analysis_result_creation(self):
        """Test AnalysisResult can be created."""
        from src.state.schema import AnalysisResult, SignalData

        result = AnalysisResult(
            ticker="AAPL",
            decision="BUY",
            confidence=80,
            position_size="Moderate",
            technical_signal=SignalData(signal="Bullish", confidence=0.75),
            fundamental_signal=SignalData(signal="Bullish", confidence=0.70),
            sentiment_signal=SignalData(signal="Neutral", confidence=0.55),
        )

        assert result.ticker == "AAPL"
        assert result.decision == "BUY"
        assert result.confidence == 80
        assert result.technical_signal.signal == "Bullish"

    def test_analysis_result_has_timestamp(self):
        """Test AnalysisResult has auto-generated timestamp."""
        from src.state.schema import AnalysisResult, SignalData

        result = AnalysisResult(
            ticker="TSLA",
            decision="HOLD",
            confidence=50,
            position_size="None",
            technical_signal=SignalData(),
            fundamental_signal=SignalData(),
            sentiment_signal=SignalData(),
        )

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestAgentState:
    """Test AgentState model."""

    def test_agent_state_required_ticker(self):
        """Test AgentState requires ticker."""
        from src.state.schema import AgentState

        state = AgentState(ticker="AAPL")

        assert state.ticker == "AAPL"

    def test_agent_state_defaults(self):
        """Test AgentState has correct defaults."""
        from src.state.schema import AgentState

        state = AgentState(ticker="AAPL")

        assert state.analysis_depth == "standard"
        assert state.price_data == {}
        assert state.company_info == {}
        assert state.technical_signal == {}
        assert state.fundamental_signal == {}
        assert state.sentiment_signal == {}
        assert state.iteration_count == 0
        assert state.max_iterations == 5
        assert state.status == "pending"
        assert state.requires_human_review is False

    def test_agent_state_to_langgraph_state(self):
        """Test AgentState conversion to dict."""
        from src.state.schema import AgentState

        state = AgentState(ticker="AAPL")
        state_dict = state.to_langgraph_state()

        assert isinstance(state_dict, dict)
        assert state_dict["ticker"] == "AAPL"
        assert "price_data" in state_dict
        assert "technical_signal" in state_dict

    def test_agent_state_from_langgraph_state(self, sample_agent_state):
        """Test AgentState creation from dict."""
        from src.state.schema import AgentState

        state = AgentState.from_langgraph_state(sample_agent_state)

        assert state.ticker == "AAPL"
        assert state.status == "completed"
        assert state.technical_signal["signal"] == "Bullish"

    def test_agent_state_to_result(self, sample_agent_state):
        """Test AgentState conversion to AnalysisResult."""
        from src.state.schema import AgentState

        state = AgentState.from_langgraph_state(sample_agent_state)
        result = state.to_result(model_used="llama3.2")

        assert result.ticker == "AAPL"
        assert result.decision == "BUY"
        assert result.confidence == 72
        assert result.model_used == "llama3.2"

    def test_agent_state_error_tracking(self):
        """Test AgentState error list."""
        from src.state.schema import AgentState

        state = AgentState(ticker="AAPL", errors=["Error 1", "Error 2"])

        assert len(state.errors) == 2
        assert "Error 1" in state.errors
