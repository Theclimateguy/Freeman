"""Knowledge-graph export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from freeman.memory.knowledgegraph import KnowledgeGraph


class KnowledgeGraphExporter:
    """Export a knowledge graph in interface-facing formats."""

    def export_html(self, knowledge_graph: KnowledgeGraph, path: str | Path) -> Path:
        result = knowledge_graph.export_html(path)
        return result if isinstance(result, Path) else Path(path).resolve()

    def export_dot(self, knowledge_graph: KnowledgeGraph, path: str | Path) -> Path:
        result = knowledge_graph.export_dot(path)
        return result if isinstance(result, Path) else Path(path).resolve()

    def export_json_ld(self, knowledge_graph: KnowledgeGraph, path: str | Path) -> Path:
        """Export nodes and edges as a simple JSON-LD graph."""

        payload: Dict[str, Any] = {
            "@context": {
                "@vocab": "https://schema.org/",
                "confidence": "https://schema.org/Float",
                "relationType": "https://schema.org/Text",
                "status": "https://schema.org/Text",
            },
            "@graph": [],
        }

        for node in knowledge_graph.nodes():
            payload["@graph"].append(
                {
                    "@id": node.id,
                    "@type": node.node_type,
                    "name": node.label,
                    "description": node.content,
                    "confidence": node.confidence,
                    "status": node.status,
                    "evidence": node.evidence,
                    "citation": node.sources,
                    "metadata": node.metadata,
                }
            )

        for edge in knowledge_graph.edges():
            payload["@graph"].append(
                {
                    "@id": edge.id,
                    "@type": "Relation",
                    "source": edge.source,
                    "target": edge.target,
                    "relationType": edge.relation_type,
                    "confidence": edge.confidence,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                }
            )

        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


__all__ = ["KnowledgeGraphExporter"]
