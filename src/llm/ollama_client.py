
from functools import lru_cache
from typing import Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config.settings import get_settings


class OllamaClient:
    """Client for interacting with Ollama models."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url
        self.llm_model = llm_model or settings.ollama_llm_model
        self.embedding_model = embedding_model or settings.ollama_embedding_model
        self.temperature = temperature or settings.agent_temperature

        self._llm: Optional[ChatOllama] = None
        self._embeddings: Optional[OllamaEmbeddings] = None

    def get_llm(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatOllama:
        """Get ChatOllama instance for the specified model."""
        model = model or self.llm_model
        temp = temperature if temperature is not None else self.temperature

        return ChatOllama(
            base_url=self.base_url,
            model=model,
            temperature=temp,
        )

    def get_embeddings(
        self,
        model: Optional[str] = None,
    ) -> OllamaEmbeddings:
        """Get OllamaEmbeddings instance for the specified model."""
        model = model or self.embedding_model

        return OllamaEmbeddings(
            base_url=self.base_url,
            model=model,
        )

    @property
    def llm(self) -> ChatOllama:
        """Get default LLM instance (cached)."""
        if self._llm is None:
            self._llm = self.get_llm()
        return self._llm

    @property
    def embeddings(self) -> OllamaEmbeddings:
        """Get default embeddings instance (cached)."""
        if self._embeddings is None:
            self._embeddings = self.get_embeddings()
        return self._embeddings


@lru_cache
def get_ollama_client() -> OllamaClient:
    """Get cached OllamaClient instance."""
    return OllamaClient()
