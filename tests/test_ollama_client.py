"""Tests for src/llm/ollama_client.py module."""

from unittest.mock import patch, MagicMock

import pytest


class TestOllamaClient:
    """Test OllamaClient class."""

    def test_ollama_client_initialization_defaults(self, mock_ollama_client):
        """Test OllamaClient initializes with defaults from settings."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()

        assert client.base_url == "http://localhost:11434"
        assert client.llm_model == "llama3.2"
        assert client.embedding_model == "nomic-embed-text"

    def test_ollama_client_initialization_custom(self, mock_ollama_client):
        """Test OllamaClient initializes with custom values."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient(
            base_url="http://custom:11434",
            llm_model="mistral",
            embedding_model="all-minilm",
            temperature=0.5,
        )

        assert client.base_url == "http://custom:11434"
        assert client.llm_model == "mistral"
        assert client.embedding_model == "all-minilm"
        assert client.temperature == 0.5

    def test_get_llm_returns_chat_ollama(self, mock_ollama_client):
        """Test get_llm returns ChatOllama instance."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()
        llm = client.get_llm()

        # Verify ChatOllama was called
        mock_ollama_client["chat"].assert_called()
        assert llm is not None

    def test_get_llm_with_custom_model(self, mock_ollama_client):
        """Test get_llm with custom model."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()
        llm = client.get_llm(model="codellama", temperature=0.2)

        # Verify ChatOllama was called with custom params
        mock_ollama_client["chat"].assert_called_with(
            base_url="http://localhost:11434",
            model="codellama",
            temperature=0.2,
        )

    def test_get_embeddings_returns_ollama_embeddings(self, mock_ollama_client):
        """Test get_embeddings returns OllamaEmbeddings instance."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()
        embeddings = client.get_embeddings()

        # Verify OllamaEmbeddings was called
        mock_ollama_client["embeddings"].assert_called()
        assert embeddings is not None

    def test_get_embeddings_with_custom_model(self, mock_ollama_client):
        """Test get_embeddings with custom model."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()
        embeddings = client.get_embeddings(model="mxbai-embed-large")

        # Verify OllamaEmbeddings was called with custom params
        mock_ollama_client["embeddings"].assert_called_with(
            base_url="http://localhost:11434",
            model="mxbai-embed-large",
        )

    def test_llm_property_caching(self, mock_ollama_client):
        """Test llm property returns cached instance."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()

        # First access creates instance
        llm1 = client.llm
        # Second access returns same instance
        llm2 = client.llm

        assert llm1 is llm2

    def test_embeddings_property_caching(self, mock_ollama_client):
        """Test embeddings property returns cached instance."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()

        # First access creates instance
        embed1 = client.embeddings
        # Second access returns same instance
        embed2 = client.embeddings

        assert embed1 is embed2


class TestGetOllamaClient:
    """Test get_ollama_client function."""

    def test_get_ollama_client_returns_instance(self, mock_ollama_client):
        """Test get_ollama_client returns OllamaClient."""
        from src.llm.ollama_client import get_ollama_client, OllamaClient

        # Clear cache
        get_ollama_client.cache_clear()

        client = get_ollama_client()

        assert isinstance(client, OllamaClient)

    def test_get_ollama_client_is_cached(self, mock_ollama_client):
        """Test get_ollama_client returns cached instance."""
        from src.llm.ollama_client import get_ollama_client

        # Clear cache
        get_ollama_client.cache_clear()

        client1 = get_ollama_client()
        client2 = get_ollama_client()

        assert client1 is client2
