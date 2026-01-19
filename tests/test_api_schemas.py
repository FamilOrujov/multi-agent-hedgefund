
from datetime import datetime

import pytest


class TestAnalysisDepth:
    """Test AnalysisDepth enum."""

    def test_analysis_depth_values(self):
        """Test AnalysisDepth has expected values."""
        from src.api.schemas import AnalysisDepth

        assert AnalysisDepth.QUICK == "quick"
        assert AnalysisDepth.STANDARD == "standard"
        assert AnalysisDepth.DEEP == "deep"

    def test_analysis_depth_is_string_enum(self):
        """Test AnalysisDepth is a string enum."""
        from src.api.schemas import AnalysisDepth

        assert isinstance(AnalysisDepth.QUICK.value, str)
        assert str(AnalysisDepth.QUICK.value) == "quick"


class TestAnalysisStatus:
    """Test AnalysisStatus enum."""

    def test_analysis_status_values(self):
        """Test AnalysisStatus has expected values."""
        from src.api.schemas import AnalysisStatus

        assert AnalysisStatus.PENDING == "pending"
        assert AnalysisStatus.RUNNING == "running"
        assert AnalysisStatus.COMPLETED == "completed"
        assert AnalysisStatus.FAILED == "failed"


class TestAnalysisRequest:
    """Test AnalysisRequest model."""

    def test_analysis_request_required_ticker(self):
        """Test AnalysisRequest requires ticker."""
        from src.api.schemas import AnalysisRequest

        request = AnalysisRequest(ticker="AAPL")

        assert request.ticker == "AAPL"

    def test_analysis_request_defaults(self):
        """Test AnalysisRequest has correct defaults."""
        from src.api.schemas import AnalysisRequest, AnalysisDepth

        request = AnalysisRequest(ticker="TSLA")

        assert request.depth == AnalysisDepth.STANDARD
        assert request.enable_hitl is False

    def test_analysis_request_custom_values(self):
        """Test AnalysisRequest with custom values."""
        from src.api.schemas import AnalysisRequest, AnalysisDepth

        request = AnalysisRequest(
            ticker="MSFT",
            depth=AnalysisDepth.DEEP,
            enable_hitl=True,
        )

        assert request.ticker == "MSFT"
        assert request.depth == AnalysisDepth.DEEP
        assert request.enable_hitl is True

    def test_analysis_request_json_schema(self):
        """Test AnalysisRequest has config example."""
        from src.api.schemas import AnalysisRequest

        schema = AnalysisRequest.model_json_schema()
        assert "example" in str(schema) or "properties" in schema


class TestAgentTestRequest:
    """Test AgentTestRequest model."""

    def test_agent_test_request_required_ticker(self):
        """Test AgentTestRequest requires ticker."""
        from src.api.schemas import AgentTestRequest

        request = AgentTestRequest(ticker="AAPL")

        assert request.ticker == "AAPL"
        assert request.custom_state is None

    def test_agent_test_request_with_custom_state(self):
        """Test AgentTestRequest with custom state."""
        from src.api.schemas import AgentTestRequest

        custom = {"price_data": {"close": 150.0}}
        request = AgentTestRequest(ticker="AAPL", custom_state=custom)

        assert request.custom_state == custom


class TestHealthResponse:
    """Test HealthResponse model."""

    def test_health_response_required_status(self):
        """Test HealthResponse requires status."""
        from src.api.schemas import HealthResponse

        response = HealthResponse(status="healthy")

        assert response.status == "healthy"

    def test_health_response_defaults(self):
        """Test HealthResponse has correct defaults."""
        from src.api.schemas import HealthResponse

        response = HealthResponse(status="healthy")

        assert response.version == "0.1.0"
        assert response.timestamp is not None


class TestOllamaHealthResponse:
    """Test OllamaHealthResponse model."""

    def test_ollama_health_connected(self):
        """Test OllamaHealthResponse when connected."""
        from src.api.schemas import OllamaHealthResponse

        response = OllamaHealthResponse(connected=True, model="llama3.2")

        assert response.connected is True
        assert response.model == "llama3.2"
        assert response.error is None

    def test_ollama_health_disconnected(self):
        """Test OllamaHealthResponse when disconnected."""
        from src.api.schemas import OllamaHealthResponse

        response = OllamaHealthResponse(connected=False, error="Connection refused")

        assert response.connected is False
        assert response.error == "Connection refused"


class TestAgentInfo:
    """Test AgentInfo model."""

    def test_agent_info_creation(self):
        """Test AgentInfo can be created."""
        from src.api.schemas import AgentInfo

        info = AgentInfo(
            name="technical_analyst",
            role="Technical Analyst",
            model="llama3.2",
            temperature=0.7,
        )

        assert info.name == "technical_analyst"
        assert info.role == "Technical Analyst"
        assert info.model == "llama3.2"
        assert info.temperature == 0.7


class TestAgentListResponse:
    """Test AgentListResponse model."""

    def test_agent_list_response(self):
        """Test AgentListResponse structure."""
        from src.api.schemas import AgentListResponse, AgentInfo

        agents = [
            AgentInfo(name="data_scout", role="Data Scout"),
            AgentInfo(name="technical_analyst", role="Technical Analyst"),
        ]

        response = AgentListResponse(agents=agents, count=2)

        assert len(response.agents) == 2
        assert response.count == 2


class TestAnalysisResponse:
    """Test AnalysisResponse model."""

    def test_analysis_response_required_fields(self):
        """Test AnalysisResponse required fields."""
        from src.api.schemas import AnalysisResponse, AnalysisStatus

        response = AnalysisResponse(
            id="test-123",
            ticker="AAPL",
            status=AnalysisStatus.PENDING,
        )

        assert response.id == "test-123"
        assert response.ticker == "AAPL"
        assert response.status == AnalysisStatus.PENDING

    def test_analysis_response_completed(self):
        """Test AnalysisResponse with completed status."""
        from src.api.schemas import AnalysisResponse, AnalysisStatus

        response = AnalysisResponse(
            id="test-456",
            ticker="TSLA",
            status=AnalysisStatus.COMPLETED,
            decision="BUY",
            confidence=80,
            position_size="Moderate",
            thesis="Strong technical and fundamental indicators...",
            technical_signal={"signal": "Bullish", "confidence": 0.75},
            fundamental_signal={"signal": "Bullish", "confidence": 0.70},
            sentiment_signal={"signal": "Neutral", "confidence": 0.55},
            duration_ms=45000,
        )

        assert response.decision == "BUY"
        assert response.confidence == 80
        assert response.duration_ms == 45000


class TestStreamEvent:
    """Test StreamEvent model."""

    def test_stream_event_creation(self):
        """Test StreamEvent can be created."""
        from src.api.schemas import StreamEvent

        event = StreamEvent(
            event="agent_start",
            agent="technical_analyst",
            data={"status": "analyzing"},
        )

        assert event.event == "agent_start"
        assert event.agent == "technical_analyst"
        assert event.timestamp is not None
