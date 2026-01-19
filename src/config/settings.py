
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_llm_model: str = Field(
        default="llama3.2",
        description="Default Ollama LLM model for agents",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Default Ollama embedding model",
    )

    # Database Configuration
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="postgres")
    postgres_db: str = Field(default="alpha_council")

    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        description="ChromaDB persistence directory",
    )
    chroma_collection_name: str = Field(
        default="alpha_council_docs",
        description="Default ChromaDB collection name",
    )

    # Agent Configuration
    max_iterations: int = Field(
        default=5,
        description="Maximum iterations for agent reasoning loop",
    )
    agent_temperature: float = Field(
        default=0.7,
        description="Default temperature for agent LLM calls",
    )

    # Tavily Configuration
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key for web search")

    # Langfuse Configuration (optional)
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_host: Optional[str] = Field(default=None)

    # Data Directory
    data_dir: Path = Field(
        default=Path("./data"),
        description="Base data directory",
    )

    @property
    def postgres_dsn(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_async_dsn(self) -> str:
        """Generate async PostgreSQL connection string."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
