"""Knowledge-graph export helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict

from freeman.memory.knowledgegraph import KnowledgeGraph


class KnowledgeGraphExporter:
    """Export a knowledge graph in interface-facing formats."""

    def export_html(self, knowledge_graph: KnowledgeGraph, path: str | Path) -> Path:
        payload = self._build_viewer_payload(knowledge_graph)
        html = self._render_flat_html(payload)
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return target

    def export_html_3d(self, knowledge_graph: KnowledgeGraph, path: str | Path) -> Path:
        """Export a richer interactive 3D HTML view of the knowledge graph.

        The viewer is intentionally structured around a generic payload envelope so
        it can later consume multiple time snapshots for graph-evolution playback
        without rewriting the UI shell.
        """

        payload = self._build_viewer_payload(knowledge_graph)
        html = self._render_3d_html(payload)
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return target

    def _build_viewer_payload(self, knowledge_graph: KnowledgeGraph) -> Dict[str, Any]:
        """Build the shared graph envelope used by 2D and 3D viewers."""

        nodes = [node.snapshot() for node in knowledge_graph.nodes(lazy_embed=False)]
        edges = [edge.snapshot() for edge in knowledge_graph.edges()]
        degree_by_node = {node["id"]: 0 for node in nodes}
        for edge in edges:
            degree_by_node[edge["source"]] = degree_by_node.get(edge["source"], 0) + 1
            degree_by_node[edge["target"]] = degree_by_node.get(edge["target"], 0) + 1

        for node in nodes:
            metadata = dict(node.get("metadata", {}))
            metadata["degree"] = degree_by_node.get(node["id"], 0)
            node["metadata"] = metadata

        locale = "ru" if any(str(node.get("metadata", {}).get("locale", "")).lower() == "ru" for node in nodes) else "en"

        payload: Dict[str, Any] = {
            "meta": {
                "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "locale": locale,
                "timeline_ready": False,
                "timeline_note": (
                    "Viewer envelope reserves space for future multi-snapshot graph evolution playback."
                ),
                "graph_path": str(knowledge_graph.json_path),
            },
            "graphData": {
                "nodes": nodes,
                "links": [
                    {
                        **edge,
                        "source": edge["source"],
                        "target": edge["target"],
                    }
                    for edge in edges
                ],
            },
            "snapshots": [],
        }
        return payload

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

    def _render_flat_html(self, payload: Dict[str, Any]) -> str:
        meta = dict(payload.get("meta", {}) or {})
        graph_data = dict(payload.get("graphData", {}) or {})
        raw_nodes = list(graph_data.get("nodes", []) or [])
        raw_edges = list(graph_data.get("links", []) or [])
        is_ru = str(meta.get("locale", "")).lower() == "ru"
        lang = "ru" if is_ru else "en"

        type_priority = [
            "domain",
            "resource",
            "outcome",
            "outcome_projection",
            "metric",
            "variable_state",
            "param_delta",
            "analysis_run",
            "signal_event",
            "epistemic_log",
            "active_hypothesis",
            "belief_conflict",
            "actor_profile",
            "actor_type",
            "scenario",
            "signal_type",
            "source_type",
            "force_domain",
            "uncertainty_factor",
        ]
        runtime_node_types = {
            "variable_state",
            "outcome_projection",
            "param_delta",
            "analysis_run",
            "epistemic_log",
            "active_hypothesis",
            "attention_focus",
            "goal_state",
            "identity_trait",
            "self_capability",
            "self_observation",
            "belief_conflict",
            "signal_event",
        }
        base_type_colors = {
            "domain": "#0f172a",
            "resource": "#047857",
            "outcome": "#be123c",
            "outcome_projection": "#be123c",
            "metric": "#4d7c0f",
            "variable_state": "#0f766e",
            "param_delta": "#b45309",
            "analysis_run": "#0891b2",
            "signal_event": "#1d4ed8",
            "epistemic_log": "#7c3aed",
            "active_hypothesis": "#a21caf",
            "belief_conflict": "#be185d",
            "scenario": "#0891b2",
            "actor_type": "#be185d",
            "actor_profile": "#4d7c0f",
            "signal_type": "#1d4ed8",
            "source_type": "#9333ea",
            "force_domain": "#0f766e",
            "uncertainty_factor": "#0369a1",
        }
        fallback_type_colors = [
            "#1d4ed8",
            "#15803d",
            "#b45309",
            "#0369a1",
            "#a21caf",
            "#be185d",
            "#4d7c0f",
            "#0f766e",
            "#be123c",
            "#64748b",
        ]
        base_relation_colors = {
            "supports": "#4f46e5",
            "поддерживает": "#4f46e5",
            "causes": "#dc2626",
            "вызывает": "#dc2626",
            "propagates_to": "#0891b2",
            "передается к": "#0891b2",
            "belongs_to_domain": "#15803d",
            "относится к домену": "#15803d",
            "observes": "#15803d",
            "наблюдает": "#15803d",
            "indicates": "#0f766e",
            "указывает на": "#0f766e",
            "contradicts": "#be185d",
            "противоречит": "#be185d",
            "reduces": "#059669",
            "снижает": "#059669",
            "threshold_exceeded": "#d97706",
            "порог превышен": "#d97706",
        }
        fallback_relation_colors = [
            "#0369a1",
            "#4f46e5",
            "#be185d",
            "#15803d",
            "#9333ea",
            "#b45309",
            "#0f766e",
            "#d97706",
        ]

        def display_type(node: dict[str, Any]) -> str:
            metadata = dict(node.get("metadata", {}) or {})
            if is_ru and metadata.get("node_type_ru"):
                return str(metadata.get("node_type_ru"))
            return str(node.get("node_type") or "node")

        def display_relation(edge: dict[str, Any]) -> str:
            metadata = dict(edge.get("metadata", {}) or {})
            if is_ru and metadata.get("relation_ru"):
                return str(metadata.get("relation_ru"))
            return str(edge.get("relation_type") or "relation")

        indegree = {str(node.get("id")): 0 for node in raw_nodes}
        outdegree = {str(node.get("id")): 0 for node in raw_nodes}
        for edge in raw_edges:
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            if source in outdegree:
                outdegree[source] += 1
            if target in indegree:
                indegree[target] += 1

        type_labels_by_raw: dict[str, str] = {}
        for node in raw_nodes:
            raw_type = str(node.get("node_type") or "node")
            type_labels_by_raw.setdefault(raw_type, display_type(node))
        sorted_raw_types = sorted(
            type_labels_by_raw,
            key=lambda value: (
                type_priority.index(value) if value in type_priority else len(type_priority),
                type_labels_by_raw[value],
            ),
        )
        group_order = [type_labels_by_raw[raw_type] for raw_type in sorted_raw_types]

        type_colors: dict[str, str] = {}
        for index, raw_type in enumerate(sorted_raw_types):
            label = type_labels_by_raw[raw_type]
            type_colors[label] = base_type_colors.get(raw_type, fallback_type_colors[index % len(fallback_type_colors)])

        relation_labels_by_raw: dict[str, str] = {}
        for edge in raw_edges:
            raw_relation = str(edge.get("relation_type") or "relation")
            relation_labels_by_raw.setdefault(raw_relation, display_relation(edge))
        sorted_raw_relations = sorted(relation_labels_by_raw, key=lambda value: relation_labels_by_raw[value])
        edge_colors: dict[str, str] = {}
        for index, raw_relation in enumerate(sorted_raw_relations):
            label = relation_labels_by_raw[raw_relation]
            edge_colors[label] = base_relation_colors.get(
                label,
                base_relation_colors.get(raw_relation, fallback_relation_colors[index % len(fallback_relation_colors)]),
            )

        flat_nodes: list[dict[str, Any]] = []
        for node in raw_nodes:
            node_id = str(node.get("id"))
            raw_type = str(node.get("node_type") or "node")
            type_label = display_type(node)
            degree = int(indegree.get(node_id, 0) + outdegree.get(node_id, 0))
            metadata = dict(node.get("metadata", {}) or {})
            metadata.pop("embedding", None)
            flat_nodes.append(
                {
                    "id": node_id,
                    "label": str(node.get("label") or node_id),
                    "type": type_label,
                    "type_key": raw_type,
                    "confidence": float(node.get("confidence") or 0.0),
                    "status": str(node.get("status") or ""),
                    "degree": degree,
                    "indegree": int(indegree.get(node_id, 0)),
                    "outdegree": int(outdegree.get(node_id, 0)),
                    "content": str(node.get("content") or ""),
                    "metadata": metadata,
                    "sources": list(node.get("sources") or []),
                    "color": metadata.get("visual_color") or type_colors.get(type_label, "#64748b"),
                }
            )

        flat_edges: list[dict[str, Any]] = []
        for edge in raw_edges:
            relation_label = display_relation(edge)
            flat_edges.append(
                {
                    "id": str(edge.get("id") or f"{edge.get('source')}:{edge.get('target')}"),
                    "source": str(edge.get("source")),
                    "target": str(edge.get("target")),
                    "relation": relation_label,
                    "relation_key": str(edge.get("relation_type") or "relation"),
                    "confidence": float(edge.get("confidence") or 0.0),
                    "weight": float(edge.get("weight") or 0.0),
                    "metadata": dict(edge.get("metadata", {}) or {}),
                    "color": edge_colors.get(relation_label, "#4f46e5"),
                }
            )

        type_counts = {
            type_label: sum(1 for node in flat_nodes if node["type"] == type_label)
            for type_label in group_order
        }
        relation_counts = {
            relation_labels_by_raw[raw_relation]: sum(
                1 for edge in flat_edges if edge["relation"] == relation_labels_by_raw[raw_relation]
            )
            for raw_relation in sorted_raw_relations
        }
        top_nodes = [
            {
                "id": node["id"],
                "label": node["label"],
                "node_type": node["type"],
                "degree": node["degree"],
                "indegree": node["indegree"],
                "outdegree": node["outdegree"],
            }
            for node in sorted(flat_nodes, key=lambda item: (-int(item["degree"]), item["id"]))[:20]
        ]
        stats = {
            "node_count": len(flat_nodes),
            "edge_count": len(flat_edges),
            "node_types": type_counts,
            "relations": relation_counts,
            "top_nodes": top_nodes,
        }

        core_types = [
            type_labels_by_raw[raw_type]
            for raw_type in sorted_raw_types
            if raw_type not in runtime_node_types
        ] or group_order
        graph = {
            "nodes": flat_nodes,
            "edges": flat_edges,
            "groupOrder": group_order,
            "typeColors": type_colors,
            "edgeColors": edge_colors,
            "coreTypes": core_types,
            "coreRelations": list(edge_colors.keys()),
        }

        title = "Freeman: flat-граф знаний" if is_ru else "Freeman Flat Knowledge Graph"
        subtitle = (
            "Колоночное интерактивное представление текущего снимка графа знаний."
            if is_ru
            else "Columnar interactive view of the current knowledge graph snapshot."
        )
        filters_label = "Фильтры" if is_ru else "Filters"
        search_label = "Поиск" if is_ru else "Search"
        search_placeholder = "id, метка, содержание" if is_ru else "node id, label, content"
        relation_label = "Связь" if is_ru else "Relation"
        all_relations = "все связи" if is_ru else "all relations"
        core_hint = (
            "Базовый вид оставляет основные онтологические типы и скрывает runtime-слой."
            if is_ru
            else "Core view keeps ontology-level types and hides runtime layer nodes."
        )
        core_button = "Базовый вид" if is_ru else "Core view"
        all_button = "Все" if is_ru else "All"
        node_types_label = "Типы узлов" if is_ru else "Node Types"
        stats_label = "Статистика" if is_ru else "Stats"
        selection_label = "Выбор" if is_ru else "Selection"
        detail_hint = "Наведите на узел или связь." if is_ru else "Hover a node or edge."
        source_label = "источник" if is_ru else "source"
        source = str(meta.get("graph_path") or "")
        warning = (
            '<span class="chip warning">граф большой: используйте фильтры</span>'
            if is_ru and len(flat_nodes) > 600
            else '<span class="chip warning">large graph: use filters</span>' if len(flat_nodes) > 600 else ""
        )
        relation_options = "\n".join(
            f'      <option value="{escape(label)}">{escape(label)}</option>' for label in edge_colors
        )
        type_checkboxes = "\n".join(
            (
                f'      <label><input type="checkbox" class="type-filter" value="{escape(type_label)}" checked> '
                f'<span style="background:{escape(type_colors[type_label])}"></span>{escape(type_label)} '
                f'<em>{type_counts.get(type_label, 0)}</em></label>'
            )
            for type_label in group_order
        )
        stats_json = escape(json.dumps(stats, ensure_ascii=False, indent=2))
        graph_json = json.dumps(graph, ensure_ascii=False)

        html = """<!doctype html>
<html lang="__LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #101418;
  --panel: rgba(21, 27, 34, 0.88);
  --panel-strong: rgba(31, 39, 49, 0.94);
  --ink: #edf2f7;
  --muted: #a5b4c3;
  --line: rgba(148, 163, 184, 0.2);
  --field: #111820;
  --field-border: rgba(148, 163, 184, 0.32);
  --accent: #7dd3fc;
}
* { box-sizing: border-box; }
html, body { min-height: 100%; }
body {
  margin: 0;
  background: radial-gradient(circle at 18% 0%, rgba(14, 165, 233, 0.18) 0, transparent 28%),
              radial-gradient(circle at 82% 18%, rgba(20, 184, 166, 0.12) 0, transparent 30%),
              linear-gradient(135deg, #0b1015 0%, #111820 48%, #151a22 100%);
  color: var(--ink);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
}
header {
  padding: 22px 28px 14px;
  border-bottom: 1px solid var(--line);
}
h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
h2 { margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }
h3 { margin: 14px 0 8px; letter-spacing: 0; }
.summary { display: flex; flex-wrap: wrap; gap: 10px; color: var(--muted); }
.chip { background: rgba(15,23,42,0.62); border: 1px solid var(--line); padding: 5px 9px; border-radius: 999px; }
.layout { display: grid; grid-template-columns: 320px 1fr 360px; min-height: calc(100vh - 92px); }
aside, .detail { background: var(--panel); backdrop-filter: blur(10px); border-right: 1px solid var(--line); padding: 16px; overflow: auto; }
.detail { border-right: 0; border-left: 1px solid var(--line); }
.canvas-wrap { overflow: auto; position: relative; }
.controls input[type="search"], .controls select {
  width: 100%; padding: 9px 10px; margin: 5px 0 12px; color: var(--ink); border: 1px solid var(--field-border); border-radius: 10px; background: var(--field);
}
.type-list { display: grid; gap: 6px; max-height: 300px; overflow: auto; }
.type-list label { display: flex; align-items: center; gap: 7px; }
.type-list span { display: inline-block; width: 13px; height: 13px; border-radius: 4px; flex: 0 0 auto; }
.type-list em { margin-left: auto; color: var(--muted); font-style: normal; font-size: 12px; }
button { color: var(--ink); border: 1px solid var(--field-border); border-radius: 10px; background: var(--panel-strong); padding: 8px 10px; cursor: pointer; margin-right: 6px; margin-bottom: 8px; }
button:hover { border-color: rgba(125, 211, 252, 0.7); }
svg { background: rgba(10,15,20,0.35); min-width: 1900px; }
.edge { fill: none; stroke-width: 1.1; opacity: 0.28; }
.edge.core { stroke-width: 2.2; opacity: 0.72; }
.edge.dim { opacity: 0.04; }
.node rect { stroke-width: 1.15; }
.node text { fill: #e5edf6; font: 500 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "Noto Sans", sans-serif; pointer-events: none; }
.node.dim { opacity: 0.12; }
.node:hover rect { stroke: var(--accent); stroke-width: 1.8; filter: drop-shadow(0 0 8px rgba(125,211,252,0.18)); }
.node.focused rect { stroke: var(--accent); stroke-width: 1.8; filter: drop-shadow(0 0 8px rgba(125,211,252,0.18)); }
.node.pinned rect { stroke: #f8fafc; stroke-width: 2.1; filter: drop-shadow(0 0 10px rgba(248,250,252,0.2)); }
.col-title { font: 700 13px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "Noto Sans", sans-serif; fill: #cbd5e1; }
pre { white-space: pre-wrap; word-break: break-word; color: #dbeafe; background: rgba(15,23,42,0.72); padding: 10px; border-radius: 10px; border: 1px solid var(--line); }
.small { color: var(--muted); font-size: 12px; }
.warning { border-color: #f59e0b; color: #fbbf24; }
@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; }
  aside, .detail { border: 0; border-bottom: 1px solid var(--line); max-height: 42vh; }
}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="summary">
    <span class="chip">__NODE_COUNT__ nodes</span>
    <span class="chip">__EDGE_COUNT__ edges</span>
    <span class="chip">__SOURCE_LABEL__: __SOURCE__</span>
    __WARNING__
  </div>
</header>
<div class="layout">
  <aside class="controls">
    <h2>__FILTERS_LABEL__</h2>
    <label>__SEARCH_LABEL__</label>
    <input id="search" type="search" placeholder="__SEARCH_PLACEHOLDER__">
    <label>__RELATION_LABEL__</label>
    <select id="relation">
      <option value="">__ALL_RELATIONS__</option>
__RELATION_OPTIONS__
    </select>
    <p class="small">__CORE_HINT__</p>
    <button id="core">__CORE_BUTTON__</button>
    <button id="all">__ALL_BUTTON__</button>
    <h3>__NODE_TYPES_LABEL__</h3>
    <div class="type-list">
__TYPE_CHECKBOXES__
    </div>
    <h3>__STATS_LABEL__</h3>
    <pre>__STATS_JSON__</pre>
  </aside>
  <main class="canvas-wrap"><svg id="graph"></svg></main>
  <section class="detail">
    <h2>__SELECTION_LABEL__</h2>
    <div id="detail">__DETAIL_HINT__</div>
  </section>
</div>
<script>
const graph = __GRAPH_JSON__;
const ui = __UI_JSON__;
const coreTypes = new Set(graph.coreTypes || []);
const coreRelations = new Set(graph.coreRelations || []);
const svg = document.getElementById('graph');
const detail = document.getElementById('detail');
const search = document.getElementById('search');
const relation = document.getElementById('relation');
const typeInputs = [...document.querySelectorAll('.type-filter')];
const nodeById = new Map(graph.nodes.map(n => [n.id, n]));
let coreOnly = false;
let pinnedNodeId = null;
const adjacency = new Map();
for (const n of graph.nodes) adjacency.set(n.id, new Set());
for (const e of graph.edges) {
  if (adjacency.has(e.source)) adjacency.get(e.source).add(e.target);
  if (adjacency.has(e.target)) adjacency.get(e.target).add(e.source);
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[ch]));
}

function truncateLabel(value) {
  const text = String(value ?? '');
  return text.length > 20 ? `${text.slice(0, 19)}…` : text;
}

function activeTypes() {
  return new Set(typeInputs.filter(i => i.checked).map(i => i.value));
}

function clearFocus() {
  document.querySelectorAll('.dim').forEach(el => el.classList.remove('dim'));
  document.querySelectorAll('.focused').forEach(el => el.classList.remove('focused'));
  document.querySelectorAll('.pinned').forEach(el => el.classList.remove('pinned'));
}

function nodeDetailHtml(n, stateLabel) {
  const state = stateLabel ? `<p class="small">${escapeHtml(stateLabel)}</p>` : '';
  const sources = Array.isArray(n.sources) && n.sources.length ? `<h3>${escapeHtml(ui.sources)}</h3><pre>${escapeHtml(n.sources.join('\\n'))}</pre>` : '';
  return `<h3>${escapeHtml(n.label)}</h3>${state}<p><b>${escapeHtml(n.id)}</b></p><p>${escapeHtml(ui.type)}=${escapeHtml(n.type)}, ${escapeHtml(ui.confidence)}=${escapeHtml(n.confidence)}, ${escapeHtml(ui.degree)}=${escapeHtml(n.degree)}</p><pre>${escapeHtml(n.content || '')}</pre>${sources}<h3>Metadata</h3><pre>${escapeHtml(JSON.stringify(n.metadata || {}, null, 2))}</pre>`;
}

function applyNodeFocus(nodeId, pinned) {
  const n = nodeById.get(nodeId);
  const selected = [...document.querySelectorAll('.node')].find(el => el.dataset.id === nodeId);
  if (!n || !selected) return false;
  clearFocus();
  selected.classList.add(pinned ? 'pinned' : 'focused');
  const neighbors = adjacency.get(nodeId) || new Set();
  document.querySelectorAll('.node').forEach(el => {
    if (el.dataset.id !== nodeId && !neighbors.has(el.dataset.id)) el.classList.add('dim');
  });
  document.querySelectorAll('.edge').forEach(el => {
    if (el.dataset.source !== nodeId && el.dataset.target !== nodeId) el.classList.add('dim');
  });
  detail.innerHTML = nodeDetailHtml(n, pinned ? ui.pinnedSelection : ui.hoverPreview);
  return true;
}

function restorePinnedFocus() {
  if (pinnedNodeId && applyNodeFocus(pinnedNodeId, true)) return;
  if (pinnedNodeId) {
    pinnedNodeId = null;
    detail.innerHTML = ui.detailHint;
  }
  clearFocus();
}

function clearPinnedSelection() {
  pinnedNodeId = null;
  clearFocus();
  detail.innerHTML = ui.detailHint;
}

function layoutNodes(nodes) {
  const groups = new Map();
  for (const n of nodes) {
    if (!groups.has(n.type)) groups.set(n.type, []);
    groups.get(n.type).push(n);
  }
  const colWidth = 172, rowHeight = 33, marginX = 34, marginY = 64;
  let maxRows = 1;
  const orderedTypes = graph.groupOrder.filter(t => groups.has(t));
  orderedTypes.forEach((type, col) => {
    const items = groups.get(type).sort((a,b) => b.degree - a.degree || a.id.localeCompare(b.id));
    maxRows = Math.max(maxRows, items.length);
    items.forEach((n, row) => {
      n.x = marginX + col * colWidth;
      n.y = marginY + row * rowHeight;
    });
  });
  return {
    width: marginX * 2 + orderedTypes.length * colWidth + 160,
    height: marginY * 2 + maxRows * rowHeight,
    orderedTypes,
    colWidth,
    marginX
  };
}

function render() {
  const q = search.value.trim().toLowerCase();
  const active = activeTypes();
  const rel = relation.value;
  const visibleNodes = graph.nodes.filter(n => active.has(n.type) && (!q || `${n.id} ${n.label} ${n.content} ${n.type}`.toLowerCase().includes(q)));
  const visibleIds = new Set(visibleNodes.map(n => n.id));
  const visibleEdges = graph.edges.filter(e => visibleIds.has(e.source) && visibleIds.has(e.target) && (!rel || e.relation === rel) && (!coreOnly || coreRelations.size === 0 || coreRelations.has(e.relation)));
  const dims = layoutNodes(visibleNodes);
  svg.setAttribute('width', dims.width);
  svg.setAttribute('height', dims.height);
  svg.innerHTML = '';
  const ns = 'http://www.w3.org/2000/svg';
  const boxWidth = 138;
  const boxHeight = 23;
  for (const [i, type] of dims.orderedTypes.entries()) {
    const text = document.createElementNS(ns, 'text');
    text.setAttribute('x', dims.marginX + i * dims.colWidth);
    text.setAttribute('y', 32);
    text.setAttribute('class', 'col-title');
    text.textContent = `${type} (${visibleNodes.filter(n => n.type === type).length})`;
    svg.appendChild(text);
  }
  const edgeLayer = document.createElementNS(ns, 'g');
  svg.appendChild(edgeLayer);
  const nodeLayer = document.createElementNS(ns, 'g');
  svg.appendChild(nodeLayer);
  for (const e of visibleEdges) {
    const s = nodeById.get(e.source), t = nodeById.get(e.target);
    if (!s || !t || s.x === undefined || t.x === undefined) continue;
    const path = document.createElementNS(ns, 'path');
    const x1 = s.x + boxWidth, y1 = s.y + boxHeight / 2, x2 = t.x, y2 = t.y + boxHeight / 2;
    const curve = Math.max(80, Math.min(220, Math.abs(x2 - x1) / 2));
    path.setAttribute('d', `M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`);
    path.setAttribute('stroke', e.color);
    path.setAttribute('class', `edge ${coreRelations.has(e.relation) ? 'core' : ''}`);
    path.dataset.source = e.source;
    path.dataset.target = e.target;
    path.dataset.relation = e.relation;
    path.addEventListener('mouseenter', () => {
      detail.innerHTML = `<h3>${escapeHtml(ui.edge)}</h3><p><b>${escapeHtml(e.source)}</b> → <b>${escapeHtml(e.target)}</b></p><p>${escapeHtml(e.relation)}</p><p>${escapeHtml(ui.confidence)}=${escapeHtml(e.confidence)}, weight=${escapeHtml(e.weight)}</p><pre>${escapeHtml(JSON.stringify(e.metadata || {}, null, 2))}</pre>`;
    });
    path.addEventListener('mouseleave', () => {
      restorePinnedFocus();
    });
    edgeLayer.appendChild(path);
  }
  for (const n of visibleNodes) {
    const g = document.createElementNS(ns, 'g');
    g.setAttribute('class', 'node');
    g.setAttribute('transform', `translate(${n.x},${n.y})`);
    g.dataset.id = n.id;
    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('width', boxWidth);
    rect.setAttribute('height', boxHeight);
    rect.setAttribute('rx', 7);
    rect.setAttribute('fill', n.color);
    rect.setAttribute('fill-opacity', ['technical_chunk', 'source'].includes(n.type_key) ? 0.34 : 0.52);
    rect.setAttribute('stroke', n.color);
    rect.setAttribute('stroke-opacity', ['technical_chunk', 'source'].includes(n.type_key) ? 0.52 : 0.78);
    const text = document.createElementNS(ns, 'text');
    text.setAttribute('x', 8);
    text.setAttribute('y', 15);
    text.textContent = truncateLabel(n.label);
    g.appendChild(rect);
    g.appendChild(text);
    g.addEventListener('mouseenter', () => {
      applyNodeFocus(n.id, false);
    });
    g.addEventListener('mouseleave', () => {
      restorePinnedFocus();
    });
    g.addEventListener('click', event => {
      event.stopPropagation();
      pinnedNodeId = n.id;
      applyNodeFocus(n.id, true);
    });
    nodeLayer.appendChild(g);
  }
  restorePinnedFocus();
}

search.addEventListener('input', render);
relation.addEventListener('change', () => {
  coreOnly = false;
  render();
});
typeInputs.forEach(input => input.addEventListener('change', () => {
  coreOnly = false;
  render();
}));
document.getElementById('core').addEventListener('click', () => {
  typeInputs.forEach(i => i.checked = coreTypes.has(i.value));
  relation.value = '';
  coreOnly = true;
  render();
});
document.getElementById('all').addEventListener('click', () => {
  typeInputs.forEach(i => i.checked = true);
  relation.value = '';
  coreOnly = false;
  render();
});
svg.addEventListener('click', event => {
  if (event.target === svg || event.target.classList.contains('col-title')) clearPinnedSelection();
});
render();
</script>
</body>
</html>"""
        replacements = {
            "__LANG__": lang,
            "__TITLE__": escape(title),
            "__NODE_COUNT__": str(len(flat_nodes)),
            "__EDGE_COUNT__": str(len(flat_edges)),
            "__SOURCE_LABEL__": escape(source_label),
            "__SOURCE__": escape(source),
            "__WARNING__": warning,
            "__FILTERS_LABEL__": escape(filters_label),
            "__SEARCH_LABEL__": escape(search_label),
            "__SEARCH_PLACEHOLDER__": escape(search_placeholder),
            "__RELATION_LABEL__": escape(relation_label),
            "__ALL_RELATIONS__": escape(all_relations),
            "__RELATION_OPTIONS__": relation_options,
            "__CORE_HINT__": escape(core_hint),
            "__CORE_BUTTON__": escape(core_button),
            "__ALL_BUTTON__": escape(all_button),
            "__NODE_TYPES_LABEL__": escape(node_types_label),
            "__TYPE_CHECKBOXES__": type_checkboxes,
            "__STATS_LABEL__": escape(stats_label),
            "__STATS_JSON__": stats_json,
            "__SELECTION_LABEL__": escape(selection_label),
            "__DETAIL_HINT__": escape(detail_hint),
            "__GRAPH_JSON__": graph_json,
            "__UI_JSON__": json.dumps(
                {
                    "detailHint": detail_hint,
                    "sources": "Источники" if is_ru else "Sources",
                    "type": "тип" if is_ru else "type",
                    "confidence": "уверенность" if is_ru else "confidence",
                    "degree": "степень" if is_ru else "degree",
                    "pinnedSelection": "Закрепленный выбор" if is_ru else "Pinned selection",
                    "hoverPreview": "Просмотр при наведении" if is_ru else "Hover preview",
                    "edge": "Связь" if is_ru else "Edge",
                },
                ensure_ascii=False,
            ),
        }
        for marker, value in replacements.items():
            html = html.replace(marker, value)
        return html

    def _render_3d_html(self, payload: Dict[str, Any]) -> str:
        data_json = json.dumps(payload, ensure_ascii=False)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Freeman 3D Knowledge Graph</title>
  <style>
    :root {{
      --bg-0: #08111f;
      --bg-1: #0f1d32;
      --panel: rgba(9, 18, 31, 0.82);
      --panel-border: rgba(140, 181, 255, 0.18);
      --text: #ecf3ff;
      --muted: #97a8c6;
      --accent: #79d2ff;
      --accent-2: #7af0c4;
      --warning: #ffba6b;
      --danger: #ff7e87;
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      height: 100%;
      overflow: hidden;
      background:
        radial-gradient(circle at 20% 15%, rgba(71, 118, 255, 0.24), transparent 22%),
        radial-gradient(circle at 80% 20%, rgba(0, 220, 170, 0.16), transparent 24%),
        radial-gradient(circle at 50% 85%, rgba(255, 170, 86, 0.12), transparent 28%),
        linear-gradient(180deg, var(--bg-0), var(--bg-1));
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #scene {{
      position: absolute;
      inset: 0;
      cursor: grab;
      touch-action: none;
    }}
    #scene canvas {{
      display: block;
      outline: none;
    }}
    .panel {{
      position: absolute;
      z-index: 5;
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .hud {{
      top: 20px;
      left: 20px;
      width: 340px;
      padding: 18px 18px 16px;
    }}
    .detail {{
      top: 20px;
      right: 20px;
      width: 360px;
      max-height: calc(100% - 40px);
      padding: 18px;
      overflow: auto;
    }}
    .footer {{
      left: 20px;
      bottom: 20px;
      width: min(720px, calc(100% - 40px));
      padding: 12px 16px;
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.15;
      letter-spacing: 0.01em;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 15px;
      color: var(--accent);
    }}
    p, .small {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 13px;
    }}
    .stats {{
      margin: 14px 0 16px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .card {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .card strong {{
      display: block;
      font-size: 20px;
      color: var(--text);
      margin-bottom: 4px;
    }}
    .controls {{
      display: grid;
      gap: 10px;
    }}
    .field {{
      display: grid;
      gap: 6px;
    }}
    .field label {{
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .field input,
    .field select {{
      width: 100%;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.10);
      border-radius: 12px;
      padding: 10px 12px;
      outline: none;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--text);
      font-size: 12px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    .detail-section {{
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .detail-grid {{
      display: grid;
      gap: 8px;
    }}
    .kv {{
      display: grid;
      gap: 3px;
    }}
    .kv span {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .kv strong {{
      font-size: 14px;
      font-weight: 600;
      color: var(--text);
      word-break: break-word;
    }}
    .list {{
      margin: 8px 0 0;
      padding-left: 18px;
      color: var(--text);
    }}
    .list li {{
      margin-bottom: 6px;
      line-height: 1.35;
      font-size: 13px;
    }}
    .footer .small {{
      color: #b7c7e1;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      background: rgba(121, 210, 255, 0.12);
      border: 1px solid rgba(121, 210, 255, 0.20);
      color: var(--accent);
    }}
    @media (max-width: 1180px) {{
      .detail {{
        width: 320px;
      }}
      .hud {{
        width: 300px;
      }}
    }}
    @media (max-width: 920px) {{
      .hud, .detail {{
        position: static;
        width: auto;
        max-height: none;
      }}
      body {{
        overflow: auto;
      }}
      #scene {{
        position: relative;
        min-height: 72vh;
      }}
      .footer {{
        position: static;
        width: auto;
        margin: 20px;
      }}
      .stack {{
        position: relative;
        z-index: 5;
        display: grid;
        gap: 14px;
        padding: 20px;
      }}
    }}
    .stack {{
      pointer-events: none;
    }}
    .stack > .panel,
    .footer {{
      pointer-events: auto;
    }}
  </style>
</head>
<body>
  <div id="scene"></div>
  <div class="stack">
    <section class="panel hud">
      <h1 id="ui-title">Freeman 3D Knowledge Graph</h1>
      <p id="ui-subtitle">Interactive 3D view of the current knowledge graph snapshot. Layout uses semantic buckets, node confidence, and graph degree.</p>
      <div class="stats">
        <div class="card"><strong id="stat-nodes">0</strong><span class="small" id="ui-nodes-label">Nodes</span></div>
        <div class="card"><strong id="stat-links">0</strong><span class="small" id="ui-links-label">Links</span></div>
        <div class="card"><strong id="stat-visible">0</strong><span class="small" id="ui-visible-label">Visible</span></div>
      </div>
      <div class="controls">
        <div class="field">
          <label for="search" id="ui-search-label">Search</label>
          <input id="search" type="text" placeholder="label, content, bucket, relation">
        </div>
        <div class="field">
          <label for="bucket" id="ui-bucket-label">Bucket</label>
          <select id="bucket"></select>
        </div>
        <div class="field">
          <label for="nodeType" id="ui-node-type-label">Node Type</label>
          <select id="nodeType"></select>
        </div>
        <div class="field">
          <label for="relation" id="ui-relation-label">Relation</label>
          <select id="relation"></select>
        </div>
        <div class="field">
          <label for="confidence" id="ui-confidence-label">Minimum Confidence</label>
          <input id="confidence" type="range" min="0" max="1" step="0.05" value="0">
          <span class="small" id="confidence-value">0.00</span>
        </div>
      </div>
      <div class="chips" id="legend"></div>
    </section>

    <aside class="panel detail">
      <h2 id="ui-selection-title">Selection</h2>
      <p id="selection-blurb">Click a node to inspect content, metadata, and local connectivity.</p>
      <div id="selection"></div>
      <div class="detail-section">
        <h2 id="ui-roadmap-title">Roadmap Hook</h2>
        <p id="ui-roadmap-text">This viewer already uses a `graphData + meta + snapshots[]` envelope. The current export writes a single snapshot and leaves `snapshots[]` empty, so a timeline / evolution slider can be added later without changing the viewer shell.</p>
      </div>
    </aside>
  </div>

  <section class="panel footer">
    <div class="small" id="ui-navigation">
      <strong style="color: var(--text);">Navigation:</strong>
      drag to orbit, scroll to zoom, right-drag to pan, click node to lock focus, double-click empty space to clear.
    </div>
    <div class="small">
      <span class="pill">Current snapshot</span>
      <span id="exported-at"></span>
    </div>
  </section>

  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
  <script src="https://unpkg.com/3d-force-graph@1.76.0/dist/3d-force-graph.min.js"></script>
  <script src="https://unpkg.com/three-spritetext@1.9.6/dist/three-spritetext.min.js"></script>
  <script>
    const payload = {data_json};
    const rawGraph = payload.graphData;
    const isRu = payload.meta?.locale === 'ru';
    const uiText = isRu ? {{
      title: 'Freeman: 3D-граф знаний',
      subtitle: 'Интерактивное 3D-представление текущего снимка графа знаний. Раскладка использует смысловые группы, уверенность узлов и степень связности.',
      nodes: 'Узлы',
      links: 'Связи',
      visible: 'Видимо',
      search: 'Поиск',
      bucket: 'Группа',
      nodeType: 'Тип узла',
      relation: 'Связь',
      confidence: 'Минимальная уверенность',
      selection: 'Выбор',
      selectionBlurb: 'Нажмите на узел, чтобы посмотреть содержание и локальные связи.',
      roadmapTitle: 'Задел под динамику',
      roadmapText: 'Этот viewer использует оболочку graphData + meta + snapshots[]. Текущий экспорт записывает один снимок; позднее можно добавить timeline без смены структуры.',
      navigation: '<strong style="color: var(--text);">Навигация:</strong> перетаскивание вращает сцену, колесо масштабирует, правая кнопка сдвигает, клик фиксирует узел, клик по фону очищает выбор.',
      allBuckets: 'Все группы',
      allNodeTypes: 'Все типы узлов',
      allRelations: 'Все связи',
      type: 'Тип',
      group: 'Группа',
      status: 'Статус',
      degree: 'Степень',
      content: 'Содержание',
      outgoing: 'Исходящие связи',
      incoming: 'Входящие связи',
      sources: 'Источники',
      noOutgoing: 'Нет исходящих связей.',
      noIncoming: 'Нет входящих связей.',
      noSources: 'Нет прикрепленных источников.',
      noContent: 'Нет содержания.'
    }} : {{
      title: 'Freeman 3D Knowledge Graph',
      subtitle: 'Interactive 3D view of the current knowledge graph snapshot. Layout uses semantic buckets, node confidence, and graph degree.',
      nodes: 'Nodes',
      links: 'Links',
      visible: 'Visible',
      search: 'Search',
      bucket: 'Bucket',
      nodeType: 'Node Type',
      relation: 'Relation',
      confidence: 'Minimum Confidence',
      selection: 'Selection',
      selectionBlurb: 'Click a node to inspect content, metadata, and local connectivity.',
      roadmapTitle: 'Roadmap Hook',
      roadmapText: 'This viewer already uses a graphData + meta + snapshots[] envelope. The current export writes a single snapshot and leaves snapshots[] empty, so a timeline / evolution slider can be added later without changing the viewer shell.',
      navigation: '<strong style="color: var(--text);">Navigation:</strong> drag to orbit, scroll to zoom, right-drag to pan, click node to lock focus, double-click empty space to clear.',
      allBuckets: 'All buckets',
      allNodeTypes: 'All node types',
      allRelations: 'All relations',
      type: 'Type',
      group: 'Bucket',
      status: 'Status',
      degree: 'Degree',
      content: 'Content',
      outgoing: 'Outgoing Links',
      incoming: 'Incoming Links',
      sources: 'Sources',
      noOutgoing: 'No outgoing links.',
      noIncoming: 'No incoming links.',
      noSources: 'No attached sources.',
      noContent: 'No content.'
    }};
    const sceneEl = document.getElementById('scene');
    const selectionEl = document.getElementById('selection');
    const selectionBlurbEl = document.getElementById('selection-blurb');
    const searchEl = document.getElementById('search');
    const bucketEl = document.getElementById('bucket');
    const nodeTypeEl = document.getElementById('nodeType');
    const relationEl = document.getElementById('relation');
    const confidenceEl = document.getElementById('confidence');
    const confidenceValueEl = document.getElementById('confidence-value');
    const legendEl = document.getElementById('legend');
    const statNodesEl = document.getElementById('stat-nodes');
    const statLinksEl = document.getElementById('stat-links');
    const statVisibleEl = document.getElementById('stat-visible');
    const exportedAtEl = document.getElementById('exported-at');
    document.getElementById('ui-title').textContent = uiText.title;
    document.getElementById('ui-subtitle').textContent = uiText.subtitle;
    document.getElementById('ui-nodes-label').textContent = uiText.nodes;
    document.getElementById('ui-links-label').textContent = uiText.links;
    document.getElementById('ui-visible-label').textContent = uiText.visible;
    document.getElementById('ui-search-label').textContent = uiText.search;
    document.getElementById('ui-bucket-label').textContent = uiText.bucket;
    document.getElementById('ui-node-type-label').textContent = uiText.nodeType;
    document.getElementById('ui-relation-label').textContent = uiText.relation;
    document.getElementById('ui-confidence-label').textContent = uiText.confidence;
    document.getElementById('ui-selection-title').textContent = uiText.selection;
    document.getElementById('selection-blurb').textContent = uiText.selectionBlurb;
    document.getElementById('ui-roadmap-title').textContent = uiText.roadmapTitle;
    document.getElementById('ui-roadmap-text').textContent = uiText.roadmapText;
    document.getElementById('ui-navigation').innerHTML = uiText.navigation;

    const bucketPalette = {{
      root: '#dbe7ff',
      forcing: '#f07d62',
      response: '#ffb25c',
      hazard: '#ff6b7d',
      exposure: '#ff8b4e',
      sector: '#b58cff',
      impact: '#ffd166',
      finance: '#6ec5ff',
      policy: '#7af0c4',
      adaptation: '#4ce3a6',
      technology: '#62d0ff',
      framework: '#8fd3ff',
      metric: '#d1f278',
      actor: '#d7a6ff',
      scenario: '#9ec5ff'
    }};
    const statusGlow = {{
      active: '#ffffff',
      uncertain: '#ffd166',
      review: '#ffba6b',
      archived: '#777e8f'
    }};

    const unique = values => Array.from(new Set(values)).filter(Boolean).sort((a, b) => String(a).localeCompare(String(b)));
    rawGraph.nodes.forEach(node => {{
      node.__nodeTypeLabel = node.metadata?.node_type_ru || node.node_type;
      node.__bucketLabel = node.metadata?.bucket_ru || node.metadata?.bucket;
    }});
    rawGraph.links.forEach(link => {{
      link.__relationLabel = link.metadata?.relation_ru || link.relation_type;
    }});
    const nodeById = new Map(rawGraph.nodes.map(node => [node.id, node]));
    const outgoingById = new Map();
    const incomingById = new Map();
    rawGraph.links.forEach(link => {{
      const source = typeof link.source === 'object' ? link.source.id : link.source;
      const target = typeof link.target === 'object' ? link.target.id : link.target;
      if (!outgoingById.has(source)) outgoingById.set(source, []);
      if (!incomingById.has(target)) incomingById.set(target, []);
      outgoingById.get(source).push(link);
      incomingById.get(target).push(link);
    }});

    function populateSelect(selectEl, label, values) {{
      selectEl.innerHTML = '';
      const allOption = document.createElement('option');
      allOption.value = '';
      allOption.textContent = label;
      selectEl.appendChild(allOption);
      values.forEach(value => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        selectEl.appendChild(option);
      }});
    }}

    populateSelect(bucketEl, uiText.allBuckets, unique(rawGraph.nodes.map(node => node.__bucketLabel)));
    populateSelect(nodeTypeEl, uiText.allNodeTypes, unique(rawGraph.nodes.map(node => node.__nodeTypeLabel)));
    populateSelect(relationEl, uiText.allRelations, unique(rawGraph.links.map(link => link.__relationLabel)));

    unique(rawGraph.nodes.map(node => node.__bucketLabel)).forEach(bucket => {{
      const chip = document.createElement('div');
      chip.className = 'chip';
      chip.innerHTML = `<span class="dot" style="background:${{bucketPalette[bucket] || '#9fb3d1'}}"></span>${{bucket}}`;
      legendEl.appendChild(chip);
    }});

    statNodesEl.textContent = String(payload.meta.node_count);
    statLinksEl.textContent = String(payload.meta.edge_count);
    exportedAtEl.textContent = `Exported: ${{payload.meta.exported_at}}`;

    const Graph = ForceGraph3D()(sceneEl)
      .backgroundColor('rgba(0,0,0,0)')
      .showNavInfo(false)
      .nodeRelSize(5)
      .linkOpacity(0.22)
      .linkWidth(link => Math.max(0.4, (link.weight || 1) * 0.85))
      .linkDirectionalParticles(link => (link.__highlight ? 3 : 0))
      .linkDirectionalParticleWidth(1.8)
      .linkColor(link => link.__highlight ? '#ffffff' : 'rgba(168, 198, 255, 0.42)')
      .cooldownTicks(160)
      .d3AlphaDecay(0.024)
      .d3VelocityDecay(0.24)
      .nodeLabel(node => `${{node.label}}\\n${{node.__nodeTypeLabel}} | ${{uiText.group}}=${{node.__bucketLabel || 'n/a'}} | confidence=${{Number(node.confidence || 0).toFixed(2)}}`);

    Graph.width(sceneEl.clientWidth || window.innerWidth);
    Graph.height(sceneEl.clientHeight || window.innerHeight);
    sceneEl.tabIndex = 0;

    const chargeForce = Graph.d3Force('charge');
    if (chargeForce) chargeForce.strength(-180);
    const linkForce = Graph.d3Force('link');
    if (linkForce) linkForce.distance(link => 60 + Math.min(70, ((link.weight || 1) * 10)));
    const controls = typeof Graph.controls === 'function' ? Graph.controls() : null;
    if (controls) {{
      controls.enabled = true;
      if ('noRotate' in controls) controls.noRotate = false;
      if ('noZoom' in controls) controls.noZoom = false;
      if ('noPan' in controls) controls.noPan = false;
      if ('zoomSpeed' in controls) controls.zoomSpeed = 5.2;
      if ('rotateSpeed' in controls) controls.rotateSpeed = 1.35;
      if ('panSpeed' in controls) controls.panSpeed = 1.2;
      if ('staticMoving' in controls) controls.staticMoving = true;
      if ('dynamicDampingFactor' in controls) controls.dynamicDampingFactor = 0.12;
    }}

    const threeScene = Graph.scene();
    const ambient = new THREE.AmbientLight(0xb9d8ff, 1.35);
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
    keyLight.position.set(120, 160, 180);
    const rimLight = new THREE.PointLight(0x7af0c4, 1.6, 800);
    rimLight.position.set(-180, -90, 160);
    threeScene.add(ambient);
    threeScene.add(keyLight);
    threeScene.add(rimLight);

    const state = {{
      selectedNodeId: null,
      hoveredNodeId: null
    }};
    let activeGraphData = null;

    function colorForNode(node) {{
      return node.metadata?.visual_color || bucketPalette[node.metadata?.bucket] || '#9fb3d1';
    }}

    function activateSceneInteraction() {{
      const active = document.activeElement;
      if (active && ['INPUT', 'SELECT', 'TEXTAREA'].includes(active.tagName)) {{
        active.blur();
      }}
      sceneEl.focus({{ preventScroll: true }});
      if (controls) {{
        controls.enabled = true;
        if ('noRotate' in controls) controls.noRotate = false;
        if ('noZoom' in controls) controls.noZoom = false;
        if ('noPan' in controls) controls.noPan = false;
        if (typeof controls.update === 'function') controls.update();
      }}
    }}

    function visibleNodeIds(filters) {{
      return new Set(rawGraph.nodes.filter(node => {{
        const haystack = [
          node.id,
          node.label,
          node.content,
          node.__nodeTypeLabel,
          node.status,
          node.__bucketLabel,
          ...(node.sources || []),
          ...(node.evidence || [])
        ].join(' ').toLowerCase();
        if (filters.search && !haystack.includes(filters.search)) return false;
        if (filters.bucket && node.__bucketLabel !== filters.bucket) return false;
        if (filters.nodeType && node.__nodeTypeLabel !== filters.nodeType) return false;
        if ((node.confidence || 0) < filters.minConfidence) return false;
        return true;
      }}).map(node => node.id));
    }}

    function currentFilters() {{
      return {{
        search: searchEl.value.trim().toLowerCase(),
        bucket: bucketEl.value,
        nodeType: nodeTypeEl.value,
        relation: relationEl.value,
        minConfidence: Number(confidenceEl.value)
      }};
    }}

    function filteredGraph() {{
      const filters = currentFilters();
      const visibleNodes = visibleNodeIds(filters);
      const links = rawGraph.links.filter(link => {{
        const source = typeof link.source === 'object' ? link.source.id : link.source;
        const target = typeof link.target === 'object' ? link.target.id : link.target;
        if (!visibleNodes.has(source) || !visibleNodes.has(target)) return false;
        if (filters.relation && link.__relationLabel !== filters.relation) return false;
        return true;
      }});
      const connectedIds = new Set();
      links.forEach(link => {{
        const source = typeof link.source === 'object' ? link.source.id : link.source;
        const target = typeof link.target === 'object' ? link.target.id : link.target;
        connectedIds.add(source);
        connectedIds.add(target);
      }});
      rawGraph.nodes.forEach(node => {{
        if (visibleNodes.has(node.id) && (!filters.relation || connectedIds.has(node.id))) {{
          connectedIds.add(node.id);
        }}
      }});
      const nodes = rawGraph.nodes
        .filter(node => connectedIds.has(node.id))
        .map(node => ({{ ...node }}));
      const normalizedLinks = links.map(link => ({{ ...link }}));
      return {{ nodes, links: normalizedLinks }};
    }}

    function renderSelection(node) {{
      if (!node) {{
        selectionBlurbEl.textContent = 'Click a node to inspect content, metadata, and local connectivity.';
        selectionEl.innerHTML = '';
        return;
      }}
      selectionBlurbEl.textContent = node.content || 'No long-form content for this node.';
      const outgoing = (outgoingById.get(node.id) || []).slice(0, 8);
      const incoming = (incomingById.get(node.id) || []).slice(0, 8);
      const metadata = node.metadata || {{}};
      const sourceList = (node.sources || []).slice(0, 6).map(source => `<li>${{source}}</li>`).join('');
      const outgoingList = outgoing.map(link => {{
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;
        const target = nodeById.get(targetId);
        return `<li><strong>${{link.__relationLabel}}</strong> → ${{target?.label || targetId}}</li>`;
      }}).join('');
      const incomingList = incoming.map(link => {{
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const source = nodeById.get(sourceId);
        return `<li>${{source?.label || sourceId}} → <strong>${{link.__relationLabel}}</strong></li>`;
      }}).join('');
      selectionEl.innerHTML = `
        <div class="detail-grid">
          <div class="kv"><span>Label</span><strong>${{node.label}}</strong></div>
          <div class="kv"><span>ID</span><strong>${{node.id}}</strong></div>
          <div class="kv"><span>${{uiText.type}}</span><strong>${{node.__nodeTypeLabel}}</strong></div>
          <div class="kv"><span>${{uiText.group}}</span><strong>${{node.__bucketLabel || 'n/a'}}</strong></div>
          <div class="kv"><span>${{uiText.status}}</span><strong>${{node.status || 'n/a'}}</strong></div>
          <div class="kv"><span>Confidence</span><strong>${{Number(node.confidence || 0).toFixed(2)}}</strong></div>
          <div class="kv"><span>${{uiText.degree}}</span><strong>${{metadata.degree ?? 0}}</strong></div>
        </div>
        <div class="detail-section">
          <h2>${{uiText.content}}</h2>
          <p>${{node.content || uiText.noContent}}</p>
        </div>
        <div class="detail-section">
          <h2>${{uiText.outgoing}}</h2>
          ${{outgoingList ? `<ul class="list">${{outgoingList}}</ul>` : `<p>${{uiText.noOutgoing}}</p>`}}
        </div>
        <div class="detail-section">
          <h2>${{uiText.incoming}}</h2>
          ${{incomingList ? `<ul class="list">${{incomingList}}</ul>` : `<p>${{uiText.noIncoming}}</p>`}}
        </div>
        <div class="detail-section">
          <h2>${{uiText.sources}}</h2>
          ${{sourceList ? `<ul class="list">${{sourceList}}</ul>` : `<p>${{uiText.noSources}}</p>`}}
        </div>
      `;
    }}

    function decorateNode(node) {{
      const degree = Number(node.metadata?.degree || 0);
      const radius = 3.2 + Math.min(7.5, degree * 0.18) + Number(node.confidence || 0) * 3.5;
      const material = new THREE.MeshStandardMaterial({{
        color: colorForNode(node),
        emissive: state.selectedNodeId === node.id ? 0xffffff : statusGlow[node.status] || '#6b7280',
        emissiveIntensity: state.selectedNodeId === node.id ? 0.40 : state.hoveredNodeId === node.id ? 0.28 : 0.14,
        roughness: 0.28,
        metalness: 0.22,
        transparent: true,
        opacity: 0.96
      }});
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(radius, 18, 18), material);
      const group = new THREE.Group();
      group.add(sphere);
      const shouldLabel = state.selectedNodeId === node.id || degree >= 10;
      if (shouldLabel) {{
        const label = new SpriteText(node.label);
        label.color = '#f3f7ff';
        label.textHeight = Math.max(6, radius * 1.15);
        label.backgroundColor = 'rgba(6, 10, 18, 0.68)';
        label.padding = 3;
        label.borderRadius = 4;
        label.position.set(radius * 1.1, radius * 1.1, 0);
        group.add(label);
      }}
      return group;
    }}

    function applyHighlights(graphData) {{
      const focusIds = new Set();
      const focusLinkIds = new Set();
      if (state.selectedNodeId) {{
        focusIds.add(state.selectedNodeId);
        (outgoingById.get(state.selectedNodeId) || []).forEach(link => {{
          const target = typeof link.target === 'object' ? link.target.id : link.target;
          focusIds.add(target);
          focusLinkIds.add(link.id);
        }});
        (incomingById.get(state.selectedNodeId) || []).forEach(link => {{
          const source = typeof link.source === 'object' ? link.source.id : link.source;
          focusIds.add(source);
          focusLinkIds.add(link.id);
        }});
      }}
      graphData.nodes.forEach(node => {{
        node.__highlight = focusIds.has(node.id) || state.hoveredNodeId === node.id;
      }});
      graphData.links.forEach(link => {{
        link.__highlight = focusLinkIds.has(link.id);
      }});
    }}

    function updateGraphData() {{
      activeGraphData = filteredGraph();
      applyHighlights(activeGraphData);
      statVisibleEl.textContent = String(activeGraphData.nodes.length);
      Graph.graphData(activeGraphData);
      Graph.refresh();
    }}

    function refreshStyles() {{
      if (!activeGraphData) return;
      applyHighlights(activeGraphData);
      Graph.refresh();
    }}

    confidenceEl.addEventListener('input', () => {{
      confidenceValueEl.textContent = Number(confidenceEl.value).toFixed(2);
      updateGraphData();
    }});
    [searchEl, bucketEl, nodeTypeEl, relationEl].forEach(el => el.addEventListener('input', updateGraphData));

    Graph
      .nodeThreeObject(node => decorateNode(node))
      .nodeThreeObjectExtend(false)
      .nodeOpacity(node => node.__highlight ? 1 : 0.92)
      .linkOpacity(link => link.__highlight ? 0.8 : 0.16)
      .linkDirectionalParticles(link => link.__highlight ? 5 : 0)
      .linkDirectionalParticleSpeed(link => link.__highlight ? 0.01 : 0)
      .linkColor(link => link.__highlight ? '#e8f2ff' : 'rgba(160, 188, 255, 0.33)');

    Graph.onNodeClick(node => {{
      state.selectedNodeId = node.id;
      renderSelection(nodeById.get(node.id));
      activateSceneInteraction();
      const distance = 110;
      const distRatio = 1 + distance / Math.hypot(node.x || 1, node.y || 1, node.z || 1);
      Graph.cameraPosition(
        {{ x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio }},
        node,
        700
      );
      window.setTimeout(activateSceneInteraction, 760);
      refreshStyles();
    }});

    Graph.onNodeHover(node => {{
      state.hoveredNodeId = node ? node.id : null;
      sceneEl.style.cursor = node ? 'pointer' : 'grab';
      refreshStyles();
    }});

    Graph.onBackgroundClick(() => {{
      state.selectedNodeId = null;
      renderSelection(null);
      activateSceneInteraction();
      sceneEl.style.cursor = 'grab';
      refreshStyles();
    }});

    Graph.onLinkClick(link => {{
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      state.selectedNodeId = sourceId;
      renderSelection(nodeById.get(sourceId));
      activateSceneInteraction();
      refreshStyles();
    }});

    Graph.onEngineStop(() => {{
      sceneEl.style.cursor = state.hoveredNodeId ? 'pointer' : 'grab';
    }});

    Graph.cameraPosition({{ x: 0, y: 0, z: 340 }});
    confidenceValueEl.textContent = Number(confidenceEl.value).toFixed(2);
    renderSelection(null);
    updateGraphData();
    ['pointerdown', 'wheel', 'mouseenter'].forEach(eventName => {{
      sceneEl.addEventListener(eventName, activateSceneInteraction, {{ passive: true }});
    }});

    window.addEventListener('resize', () => {{
      Graph.width(sceneEl.clientWidth || window.innerWidth);
      Graph.height(sceneEl.clientHeight || window.innerHeight);
      activateSceneInteraction();
    }});
  </script>
</body>
</html>
"""


__all__ = ["KnowledgeGraphExporter"]
