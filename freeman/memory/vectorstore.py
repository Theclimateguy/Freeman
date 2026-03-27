"""ChromaDB-backed semantic index for knowledge-graph nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING, List

from freeman.memory.knowledgegraph import KGNode

if TYPE_CHECKING:
    from freeman.memory.knowledgegraph import KnowledgeGraph


class KGVectorStore:
    """Persistent vector store for KG nodes."""

    def __init__(
        self,
        path: str | Path,
        collection_name: str = "kg_nodes",
        *,
        client: Any | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - exercised when optional dependency is missing
            raise RuntimeError("ChromaDB is not installed. Install the 'semantic' extra to enable vector storage.") from exc

        self.path = Path(path).resolve()
        self.collection_name = collection_name
        self._client = client or chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def upsert(self, node: KGNode) -> None:
        """Insert or update one node embedding."""

        if not node.embedding:
            return
        self._collection.upsert(
            ids=[node.id],
            embeddings=[list(node.embedding)],
            documents=[node.content or node.label],
            metadatas=[
                {
                    "confidence": float(node.confidence),
                    "status": node.status,
                    "label": node.label,
                    "node_type": node.node_type,
                }
            ],
        )

    def delete(self, node_id: str) -> None:
        """Remove a node from the vector store."""

        self._collection.delete(ids=[node_id])

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 15,
        min_confidence: float = 0.0,
    ) -> List[str]:
        """Return node ids ordered by cosine similarity."""

        if not query_embedding or top_k <= 0 or self._collection.count() == 0:
            return []
        where = {"confidence": {"$gte": float(min_confidence)}} if min_confidence > 0.0 else None
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=int(top_k),
            where=where,
        )
        ids = result.get("ids", [[]])
        return [str(node_id) for node_id in ids[0]]

    def sync_from_kg(self, kg: KnowledgeGraph) -> int:
        """Bulk upsert all nodes that already have embeddings."""

        upserted = 0
        for node in kg.nodes(lazy_embed=False):
            if node.status == "archived" or not node.embedding:
                continue
            self.upsert(node)
            upserted += 1
        return upserted


__all__ = ["KGVectorStore"]
