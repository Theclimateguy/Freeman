"""Knowledge-graph export helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from freeman_librarian.memory.knowledgegraph import KnowledgeGraph


class KnowledgeGraphExporter:
    """Export a knowledge graph in interface-facing formats."""

    def export_html(self, knowledge_graph: KnowledgeGraph, path: str | Path) -> Path:
        result = knowledge_graph.export_html(path)
        return result if isinstance(result, Path) else Path(path).resolve()

    def export_html_3d(self, knowledge_graph: KnowledgeGraph, path: str | Path) -> Path:
        """Export a richer interactive 3D HTML view of the knowledge graph.

        The viewer is intentionally structured around a generic payload envelope so
        it can later consume multiple time snapshots for graph-evolution playback
        without rewriting the UI shell.
        """

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

        payload: Dict[str, Any] = {
            "meta": {
                "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "node_count": len(nodes),
                "edge_count": len(edges),
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

        html = self._render_3d_html(payload)
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return target

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
      <h1>Freeman 3D Climate Graph</h1>
      <p>Interactive 3D view of the current knowledge graph snapshot. Layout uses semantic buckets, node confidence, and graph degree.</p>
      <div class="stats">
        <div class="card"><strong id="stat-nodes">0</strong><span class="small">Nodes</span></div>
        <div class="card"><strong id="stat-links">0</strong><span class="small">Links</span></div>
        <div class="card"><strong id="stat-visible">0</strong><span class="small">Visible</span></div>
      </div>
      <div class="controls">
        <div class="field">
          <label for="search">Search</label>
          <input id="search" type="text" placeholder="label, content, bucket, relation">
        </div>
        <div class="field">
          <label for="bucket">Bucket</label>
          <select id="bucket"></select>
        </div>
        <div class="field">
          <label for="nodeType">Node Type</label>
          <select id="nodeType"></select>
        </div>
        <div class="field">
          <label for="relation">Relation</label>
          <select id="relation"></select>
        </div>
        <div class="field">
          <label for="confidence">Minimum Confidence</label>
          <input id="confidence" type="range" min="0" max="1" step="0.05" value="0">
          <span class="small" id="confidence-value">0.00</span>
        </div>
      </div>
      <div class="chips" id="legend"></div>
    </section>

    <aside class="panel detail">
      <h2>Selection</h2>
      <p id="selection-blurb">Click a node to inspect content, metadata, and local connectivity.</p>
      <div id="selection"></div>
      <div class="detail-section">
        <h2>Roadmap Hook</h2>
        <p>This viewer already uses a `graphData + meta + snapshots[]` envelope. The current export writes a single snapshot and leaves `snapshots[]` empty, so a timeline / evolution slider can be added later without changing the viewer shell.</p>
      </div>
    </aside>
  </div>

  <section class="panel footer">
    <div class="small">
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
      allOption.textContent = `All ${{label}}`;
      selectEl.appendChild(allOption);
      values.forEach(value => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        selectEl.appendChild(option);
      }});
    }}

    populateSelect(bucketEl, 'buckets', unique(rawGraph.nodes.map(node => node.metadata?.bucket)));
    populateSelect(nodeTypeEl, 'node types', unique(rawGraph.nodes.map(node => node.node_type)));
    populateSelect(relationEl, 'relations', unique(rawGraph.links.map(link => link.relation_type)));

    unique(rawGraph.nodes.map(node => node.metadata?.bucket)).forEach(bucket => {{
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
      .nodeLabel(node => `${{node.label}}\\n${{node.node_type}} | bucket=${{node.metadata?.bucket || 'n/a'}} | confidence=${{Number(node.confidence || 0).toFixed(2)}}`);

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
      return bucketPalette[node.metadata?.bucket] || '#9fb3d1';
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
          node.node_type,
          node.status,
          node.metadata?.bucket,
          ...(node.sources || []),
          ...(node.evidence || [])
        ].join(' ').toLowerCase();
        if (filters.search && !haystack.includes(filters.search)) return false;
        if (filters.bucket && node.metadata?.bucket !== filters.bucket) return false;
        if (filters.nodeType && node.node_type !== filters.nodeType) return false;
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
        if (filters.relation && link.relation_type !== filters.relation) return false;
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
        return `<li><strong>${{link.relation_type}}</strong> → ${{target?.label || targetId}}</li>`;
      }}).join('');
      const incomingList = incoming.map(link => {{
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const source = nodeById.get(sourceId);
        return `<li>${{source?.label || sourceId}} → <strong>${{link.relation_type}}</strong></li>`;
      }}).join('');
      selectionEl.innerHTML = `
        <div class="detail-grid">
          <div class="kv"><span>Label</span><strong>${{node.label}}</strong></div>
          <div class="kv"><span>ID</span><strong>${{node.id}}</strong></div>
          <div class="kv"><span>Type</span><strong>${{node.node_type}}</strong></div>
          <div class="kv"><span>Bucket</span><strong>${{metadata.bucket || 'n/a'}}</strong></div>
          <div class="kv"><span>Status</span><strong>${{node.status || 'n/a'}}</strong></div>
          <div class="kv"><span>Confidence</span><strong>${{Number(node.confidence || 0).toFixed(2)}}</strong></div>
          <div class="kv"><span>Degree</span><strong>${{metadata.degree ?? 0}}</strong></div>
        </div>
        <div class="detail-section">
          <h2>Content</h2>
          <p>${{node.content || 'No content.'}}</p>
        </div>
        <div class="detail-section">
          <h2>Outgoing Links</h2>
          ${{outgoingList ? `<ul class="list">${{outgoingList}}</ul>` : '<p>No outgoing links.</p>'}}
        </div>
        <div class="detail-section">
          <h2>Incoming Links</h2>
          ${{incomingList ? `<ul class="list">${{incomingList}}</ul>` : '<p>No incoming links.</p>'}}
        </div>
        <div class="detail-section">
          <h2>Sources</h2>
          ${{sourceList ? `<ul class="list">${{sourceList}}</ul>` : '<p>No attached sources.</p>'}}
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
