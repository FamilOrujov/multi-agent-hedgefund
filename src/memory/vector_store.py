
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config.settings import get_settings
from src.llm.ollama_client import OllamaClient


class VectorStore:
    """ChromaDB-based vector store for semantic memory."""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        settings = get_settings()
        self.ollama_client = ollama_client or OllamaClient()
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Get or create ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama."""
        embeddings_model = self.ollama_client.embeddings
        embeddings = []
        for text in texts:
            embedding = embeddings_model.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add documents to the vector store."""
        if not documents:
            return []

        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]

        if metadatas is None:
            metadatas = [{} for _ in documents]

        embeddings = self._get_embeddings(documents)

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        query_embedding = self._get_embeddings([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "id": results["ids"][0][i] if results["ids"] else "",
                })

        return formatted_results

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self._collection = None

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def add_analysis_result(
        self,
        ticker: str,
        analysis_type: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add an analysis result to the vector store for future retrieval."""
        import uuid
        from datetime import datetime

        doc_id = f"{ticker}_{analysis_type}_{uuid.uuid4().hex[:8]}"

        doc_metadata = {
            "ticker": ticker,
            "analysis_type": analysis_type,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        self.add_documents(
            documents=[content],
            metadatas=[doc_metadata],
            ids=[doc_id],
        )

        return doc_id

    def get_historical_analyses(
        self,
        ticker: str,
        analysis_type: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve historical analyses for a ticker."""
        where_filter: dict[str, Any] = {"ticker": ticker}
        if analysis_type:
            where_filter["analysis_type"] = analysis_type

        return self.search(
            query=f"Analysis for {ticker}",
            n_results=limit,
            where=where_filter,
        )
