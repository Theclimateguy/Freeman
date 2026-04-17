"""Minimal REST interface for Freeman."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any, Dict

from freeman_librarian.core.world import WorldState
from freeman_librarian.interface.modeloverride import ModelOverrideAPI
from freeman_librarian.memory.knowledgegraph import KnowledgeGraph


def _query_node_payload(node: Any) -> Dict[str, Any]:
    """Serialize a node for query responses without large embedding vectors."""

    payload = node.snapshot()
    payload["embedding"] = []
    return payload


class InterfaceAPI:
    """Read-only v0.1 API surface over the knowledge graph."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph | None = None,
        override_api: ModelOverrideAPI | None = None,
    ) -> None:
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.override_api = override_api or ModelOverrideAPI()

    def get_status(self) -> Dict[str, Any]:
        nodes = self.knowledge_graph.nodes()
        edges = self.knowledge_graph.edges()
        counts: Dict[str, int] = {}
        for node in nodes:
            counts[node.status] = counts.get(node.status, 0) + 1
        return {
            "knowledge_graph_path": str(self.knowledge_graph.json_path),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "status_counts": counts,
        }

    def post_query(
        self,
        *,
        text: str | None = None,
        status: str | None = None,
        node_type: str | None = None,
        min_confidence: float | None = None,
        semantic: bool = False,
        limit: int | None = None,
    ) -> Dict[str, Any]:
        if semantic and text:
            matches = self.knowledge_graph.semantic_query(text, top_k=max(int(limit or 15), 1))
            filtered = []
            for node in matches:
                if status is not None and node.status != status:
                    continue
                if node_type is not None and node.node_type != node_type:
                    continue
                if min_confidence is not None and node.confidence < min_confidence:
                    continue
                filtered.append(node)
            matches = filtered
        else:
            matches = self.knowledge_graph.query(
                text=text,
                status=status,
                node_type=node_type,
                min_confidence=min_confidence,
            )
            if limit is not None:
                matches = matches[: max(int(limit), 0)]
        return {
            "matches": [_query_node_payload(node) for node in matches],
            "count": len(matches),
            "semantic": bool(semantic and text),
        }

    def register_domain(self, world: WorldState, *, machine_simulation: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.override_api.register_domain(world.domain_id, world, machine_simulation=machine_simulation)
        return {"domain_id": world.domain_id, "registered": True}

    def patch_domain_params(self, domain_id: str, overrides: Dict[str, Any], *, actor: str = "human") -> Dict[str, Any]:
        return self.override_api.patch_params(domain_id, overrides, actor=actor)

    def patch_domain_edge(
        self,
        domain_id: str,
        edge_id: int | str,
        expected_sign: str,
        *,
        actor: str = "human",
    ) -> Dict[str, Any]:
        return self.override_api.patch_edge(domain_id, edge_id, expected_sign, actor=actor)

    def rerun_domain(self, domain_id: str) -> Dict[str, Any]:
        return self.override_api.rerun_domain(domain_id)

    def get_domain_diff(self, domain_id: str) -> Dict[str, Any]:
        return self.override_api.get_diff(domain_id)


def run_server(host: str = "127.0.0.1", port: int = 8000, api: InterfaceAPI | None = None) -> ThreadingHTTPServer:
    """Start a small HTTP server serving GET /status and POST /query."""

    interface_api = api or InterfaceAPI()

    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            return json.loads(raw_body.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/status":
                parts = [part for part in self.path.split("/") if part]
                if len(parts) == 3 and parts[0] == "domain" and parts[2] == "diff":
                    self._write_json(interface_api.get_domain_diff(parts[1]))
                    return
                self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._write_json(interface_api.get_status())

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/query":
                payload = self._read_json()
                self._write_json(
                    interface_api.post_query(
                        text=payload.get("text"),
                        status=payload.get("status"),
                        node_type=payload.get("node_type"),
                        min_confidence=payload.get("min_confidence"),
                        semantic=bool(payload.get("semantic", False)),
                        limit=payload.get("limit"),
                    )
                )
                return
            parts = [part for part in self.path.split("/") if part]
            if len(parts) == 3 and parts[0] == "domain" and parts[2] == "rerun":
                self._write_json(interface_api.rerun_domain(parts[1]))
                return
            self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def do_PATCH(self) -> None:  # noqa: N802
            parts = [part for part in self.path.split("/") if part]
            payload = self._read_json()
            if len(parts) == 3 and parts[0] == "domain" and parts[2] == "params":
                self._write_json(
                    interface_api.patch_domain_params(
                        parts[1],
                        payload.get("overrides", {}),
                        actor=payload.get("actor", "human"),
                    )
                )
                return
            if len(parts) == 4 and parts[0] == "domain" and parts[2] == "edges":
                self._write_json(
                    interface_api.patch_domain_edge(
                        parts[1],
                        parts[3],
                        payload["expected_sign"],
                        actor=payload.get("actor", "human"),
                    )
                )
                return
            self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, port), Handler)
    return server


__all__ = ["InterfaceAPI", "run_server"]
