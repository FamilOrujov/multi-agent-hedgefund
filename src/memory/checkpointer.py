
import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import JSONB

from src.config.settings import get_settings

Base = declarative_base()


class Checkpoint(Base):
    """SQLAlchemy model for storing graph checkpoints."""

    __tablename__ = "checkpoints"

    id = Column(String(255), primary_key=True)
    thread_id = Column(String(255), nullable=False, index=True)
    ticker = Column(String(20), nullable=True, index=True)
    step_name = Column(String(100), nullable=True)
    state = Column(JSONB, nullable=False)
    iteration = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Checkpoint(id={self.id}, thread_id={self.thread_id}, step={self.step_name})>"


class AnalysisHistory(Base):
    """SQLAlchemy model for storing completed analysis results."""

    __tablename__ = "analysis_history"

    id = Column(String(255), primary_key=True)
    ticker = Column(String(20), nullable=False, index=True)
    decision = Column(String(20), nullable=False)
    confidence = Column(Integer, nullable=False)
    position_size = Column(String(50), nullable=True)
    final_thesis = Column(Text, nullable=True)
    state_snapshot = Column(JSONB, nullable=True)
    model_used = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<AnalysisHistory(ticker={self.ticker}, decision={self.decision})>"


class PostgresCheckpointer:
    """
    PostgreSQL-based checkpointer for LangGraph state persistence.
    
    Provides:
    - State persistence at each graph node transition
    - Time-travel debugging capability
    - Analysis history storage
    """

    def __init__(self, connection_string: Optional[str] = None):
        settings = get_settings()
        self.connection_string = connection_string or settings.postgres_dsn

        self._engine = None
        self._session_factory = None

    @property
    def engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        return self._engine

    @property
    def session_factory(self):
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    def initialize(self) -> None:
        """Create database tables if they don't exist."""
        Base.metadata.create_all(self.engine)

    def save_checkpoint(
        self,
        thread_id: str,
        state: dict[str, Any],
        step_name: str = "",
    ) -> str:
        """Save a checkpoint of the current state."""
        import uuid

        checkpoint_id = f"{thread_id}_{uuid.uuid4().hex[:8]}"

        serializable_state = self._make_serializable(state)

        checkpoint = Checkpoint(
            id=checkpoint_id,
            thread_id=thread_id,
            ticker=state.get("ticker", ""),
            step_name=step_name,
            state=serializable_state,
            iteration=state.get("iteration_count", 0),
        )

        with self.session_factory() as session:
            session.add(checkpoint)
            session.commit()

        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        """Load a specific checkpoint by ID."""
        with self.session_factory() as session:
            checkpoint = session.query(Checkpoint).filter_by(id=checkpoint_id).first()
            if checkpoint:
                return checkpoint.state
        return None

    def get_latest_checkpoint(self, thread_id: str) -> Optional[dict[str, Any]]:
        """Get the most recent checkpoint for a thread."""
        with self.session_factory() as session:
            checkpoint = (
                session.query(Checkpoint)
                .filter_by(thread_id=thread_id)
                .order_by(Checkpoint.created_at.desc())
                .first()
            )
            if checkpoint:
                return checkpoint.state
        return None

    def list_checkpoints(
        self,
        thread_id: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List checkpoints with optional filtering."""
        with self.session_factory() as session:
            query = session.query(Checkpoint)

            if thread_id:
                query = query.filter_by(thread_id=thread_id)
            if ticker:
                query = query.filter_by(ticker=ticker)

            checkpoints = query.order_by(Checkpoint.created_at.desc()).limit(limit).all()

            return [
                {
                    "id": cp.id,
                    "thread_id": cp.thread_id,
                    "ticker": cp.ticker,
                    "step_name": cp.step_name,
                    "iteration": cp.iteration,
                    "created_at": cp.created_at.isoformat(),
                }
                for cp in checkpoints
            ]

    def save_analysis(
        self,
        ticker: str,
        state: dict[str, Any],
        model_used: str = "",
    ) -> str:
        """Save a completed analysis to history."""
        import uuid

        analysis_id = f"{ticker}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        manager_decision = state.get("manager_decision", {})

        analysis = AnalysisHistory(
            id=analysis_id,
            ticker=ticker,
            decision=manager_decision.get("decision", "HOLD"),
            confidence=manager_decision.get("confidence", 0),
            position_size=manager_decision.get("position_size", "None"),
            final_thesis=state.get("final_thesis", ""),
            state_snapshot=self._make_serializable(state),
            model_used=model_used,
        )

        with self.session_factory() as session:
            session.add(analysis)
            session.commit()

        return analysis_id

    def get_analysis_history(
        self,
        ticker: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get historical analyses."""
        with self.session_factory() as session:
            query = session.query(AnalysisHistory)

            if ticker:
                query = query.filter_by(ticker=ticker)

            analyses = query.order_by(AnalysisHistory.created_at.desc()).limit(limit).all()

            return [
                {
                    "id": a.id,
                    "ticker": a.ticker,
                    "decision": a.decision,
                    "confidence": a.confidence,
                    "position_size": a.position_size,
                    "model_used": a.model_used,
                    "created_at": a.created_at.isoformat(),
                }
                for a in analyses
            ]

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
