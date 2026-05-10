"""Typed runtime contracts between Freeman state layers.

These protocols are intentionally structural. They document the minimal surface
that the runtime expects without forcing inheritance across core, memory, and
agent modules.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from freeman.memory.knowledgegraph import KGEdge, KGNode, KGSemanticSearchResult


@runtime_checkable
class WorldStateContract(Protocol):
    """Executable world state consumed by the analysis and runtime layers."""

    domain_id: str
    t: int
    runtime_step: int
    metadata: dict[str, Any]

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable world snapshot."""

    def clone(self) -> "WorldStateContract":
        """Return a detached copy for simulation/update flows."""


@runtime_checkable
class KnowledgeGraphContract(Protocol):
    """Persistent memory surface used by pipeline, runtime, and query layers."""

    def add_node(self, node: KGNode) -> None:
        """Persist or update a knowledge node."""

    def get_node(self, node_id: str, *, lazy_embed: bool = True) -> KGNode | None:
        """Return a knowledge node by id."""

    def query(
        self,
        *,
        text: str | None = None,
        status: str | None = None,
        node_type: str | None = None,
        min_confidence: float | None = None,
        metadata_filters: dict[str, Any] | None = None,
        metadata_contains: dict[str, Sequence[Any] | Any] | None = None,
    ) -> list[KGNode]:
        """Return filtered graph nodes."""

    def add_edge(self, edge: KGEdge) -> None:
        """Persist a knowledge edge."""

    def edges(self) -> list[KGEdge]:
        """Return graph edges."""

    def semantic_search(
        self,
        query_text: str,
        top_k: int = 15,
        *,
        min_score: float = 0.05,
        query_embedding: list[float] | None = None,
    ) -> KGSemanticSearchResult:
        """Return semantic hits with retrieval trace."""


@runtime_checkable
class ConsciousStateContract(Protocol):
    """Serializable agent state boundary used by deterministic operators."""

    world_ref: str
    agent_role: str
    goal_state: list[str]
    attention_state: dict[str, float]
    runtime_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready consciousness snapshot."""


__all__ = [
    "ConsciousStateContract",
    "KnowledgeGraphContract",
    "WorldStateContract",
]
