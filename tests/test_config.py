
from functools import lru_cache

import pytest


class TestSettings:
    """Test Settings class."""

    def test_settings_initialization(self):
        """Test Settings class initializes with defaults."""
        from src.config.settings import Settings

        # Clear cache to get fresh instance
        settings = Settings()

        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_llm_model == "llama3.2"
        assert settings.ollama_embedding_model == "nomic-embed-text"

    def test_settings_postgres_defaults(self):
        """Test Settings PostgreSQL defaults."""
        from src.config.settings import Settings

        settings = Settings()

        assert settings.postgres_host == "localhost"
        assert settings.postgres_port == 5432
        assert settings.postgres_user == "postgres"
        assert settings.postgres_password == "postgres"

    def test_settings_postgres_dsn_property(self):
        """Test postgres_dsn property generation."""
        from src.config.settings import Settings

        settings = Settings()
        dsn = settings.postgres_dsn

        assert "postgresql://" in dsn
        assert settings.postgres_user in dsn
        assert settings.postgres_host in dsn
        assert str(settings.postgres_port) in dsn

    def test_settings_postgres_async_dsn_property(self):
        """Test postgres_async_dsn property generation."""
        from src.config.settings import Settings

        settings = Settings()
        async_dsn = settings.postgres_async_dsn

        assert "postgresql+psycopg://" in async_dsn
        assert settings.postgres_user in async_dsn

    def test_settings_agent_configuration(self):
        """Test agent configuration defaults."""
        from src.config.settings import Settings

        settings = Settings()

        assert settings.max_iterations == 5
        assert settings.agent_temperature == 0.7

    def test_settings_chromadb_configuration(self):
        """Test ChromaDB configuration."""
        from src.config.settings import Settings

        settings = Settings()

        assert settings.chroma_persist_directory == "./data/chroma"
        assert settings.chroma_collection_name == "alpha_council_docs"

    def test_get_settings_returns_settings_instance(self):
        """Test get_settings returns Settings instance."""
        from src.config.settings import get_settings

        settings = get_settings()

        assert settings is not None
        assert hasattr(settings, "ollama_base_url")
        assert hasattr(settings, "postgres_dsn")

    def test_get_settings_is_cached(self):
        """Test get_settings returns cached instance."""
        from src.config.settings import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same cached instance
        assert settings1 is settings2
