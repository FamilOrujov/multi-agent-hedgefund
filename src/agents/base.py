
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.llm.ollama_client import OllamaClient


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self._llm: Optional[ChatOllama] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name identifier."""
        pass

    @property
    @abstractmethod
    def role(self) -> str:
        """Agent role description."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for the agent."""
        pass

    @property
    def llm(self) -> ChatOllama:
        """Get the LLM instance for this agent."""
        if self._llm is None:
            self._llm = self.ollama_client.get_llm(
                model=self.model,
                temperature=self.temperature,
            )
        return self._llm

    def invoke(self, user_message: str, context: Optional[dict[str, Any]] = None) -> str:
        """Invoke the agent with a user message and optional context."""
        messages = [
            SystemMessage(content=self._build_system_prompt(context)),
            HumanMessage(content=user_message),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _build_system_prompt(self, context: Optional[dict[str, Any]] = None) -> str:
        """Build the full system prompt with optional context."""
        base_prompt = self.system_prompt

        if context:
            context_str = "\n\n## Current Context:\n"
            for key, value in context.items():
                context_str += f"- **{key}**: {value}\n"
            return base_prompt + context_str

        return base_prompt

    @abstractmethod
    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        """Perform analysis based on current state. Returns updated state."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"
