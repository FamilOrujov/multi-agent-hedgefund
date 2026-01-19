
import httpx
from dataclasses import dataclass
from typing import Optional

from src.config.settings import get_settings


@dataclass
class OllamaModel:
    """Represents an Ollama model."""

    name: str
    size: int
    digest: str
    modified_at: str

    @property
    def size_gb(self) -> float:
        """Get model size in GB."""
        return self.size / (1024**3)

    @property
    def is_embedding_model(self) -> bool:
        """Check if this is likely an embedding model."""
        embedding_keywords = ["embed", "nomic", "bge", "e5", "gte"]
        return any(kw in self.name.lower() for kw in embedding_keywords)


class ModelManager:
    """Manages Ollama model discovery and selection."""

    def __init__(self, base_url: Optional[str] = None):
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url

    def list_models(self) -> list[OllamaModel]:
        """List all available Ollama models."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            data = response.json()

            models = []
            for model_data in data.get("models", []):
                models.append(
                    OllamaModel(
                        name=model_data["name"],
                        size=model_data.get("size", 0),
                        digest=model_data.get("digest", ""),
                        modified_at=model_data.get("modified_at", ""),
                    )
                )
            return models
        except httpx.RequestError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {e}"
            ) from e

    def list_llm_models(self) -> list[OllamaModel]:
        """List models suitable for LLM tasks."""
        return [m for m in self.list_models() if not m.is_embedding_model]

    def list_embedding_models(self) -> list[OllamaModel]:
        """List models suitable for embedding tasks."""
        return [m for m in self.list_models() if m.is_embedding_model]

    def check_model_exists(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        models = self.list_models()
        return any(m.name == model_name or m.name.startswith(model_name) for m in models)

    def get_model_info(self, model_name: str) -> Optional[OllamaModel]:
        """Get info for a specific model."""
        models = self.list_models()
        for model in models:
            if model.name == model_name or model.name.startswith(model_name):
                return model
        return None

    def is_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.RequestError:
            return False
