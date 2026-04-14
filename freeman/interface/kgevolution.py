"""HTML timeline exporter for Freeman knowledge-graph evolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SnapshotFrame:
    path: Path
    meta: dict[str, Any]
    graph_data: dict[str, Any]


class KnowledgeGraphEvolutionExporter:
    """Render a standalone HTML viewer over a directory or glob of KG snapshots."""

    def export_html(self, snapshot_source: str | Path, output_path: str | Path) -> Path:
        snapshot_paths = self._resolve_snapshot_paths(snapshot_source)
        if not snapshot_paths:
            raise FileNotFoundError(f"No snapshot JSON files found for: {snapshot_source}")
        frames = self._load_frames(snapshot_paths)
        payload = {
            "meta": {
                "exported_at": _utc_now_iso(),
                "snapshot_count": len(frames),
                "source": str(snapshot_source),
                "latest_snapshot_path": str(frames[-1].path),
            },
            "snapshots": [
                {
                    "meta": frame.meta,
                    "graphData": frame.graph_data,
                }
                for frame in frames
            ],
        }
        html = self._render_html(payload)
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return target

    def _resolve_snapshot_paths(self, snapshot_source: str | Path) -> list[Path]:
        source = str(snapshot_source)
        if any(token in source for token in "*?[]"):
            paths = [Path(item).resolve() for item in glob.glob(source)]
        else:
            candidate = Path(source).expanduser()
            if candidate.is_dir():
                paths = sorted(path.resolve() for path in candidate.glob("*.json") if path.name != "manifest.json")
            elif candidate.exists():
                paths = [candidate.resolve()]
            else:
                paths = []
        return sorted([path for path in paths if path.is_file()], key=self._sort_key)

    def _sort_key(self, path: Path) -> tuple[Any, ...]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return (float("inf"), str(path))
        meta = dict(payload.get("snapshot_meta", {}))
        runtime_step = meta.get("runtime_step")
        timestamp = meta.get("timestamp") or meta.get("exported_at") or ""
        return (
            int(runtime_step) if runtime_step is not None else float("inf"),
            str(timestamp),
            str(path.name),
        )

    def _load_frames(self, snapshot_paths: list[Path]) -> list[SnapshotFrame]:
        frames: list[SnapshotFrame] = []
        previous_node_ids: set[str] = set()
        previous_edge_ids: set[str] = set()
        for index, path in enumerate(snapshot_paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            nodes = [dict(item) for item in payload.get("nodes", [])]
            edges = [dict(item) for item in payload.get("edges", [])]
            degree_by_node = {node["id"]: 0 for node in nodes}
            for edge in edges:
                degree_by_node[edge["source"]] = degree_by_node.get(edge["source"], 0) + 1
                degree_by_node[edge["target"]] = degree_by_node.get(edge["target"], 0) + 1
            for node in nodes:
                metadata = dict(node.get("metadata", {}))
                metadata["degree"] = degree_by_node.get(node["id"], 0)
                node["metadata"] = metadata

            node_ids = {node["id"] for node in nodes}
            edge_ids = {edge["id"] for edge in edges}
            added_node_ids = sorted(node_ids - previous_node_ids)
            removed_node_ids = sorted(previous_node_ids - node_ids)
            added_edge_ids = sorted(edge_ids - previous_edge_ids)
            removed_edge_ids = sorted(previous_edge_ids - edge_ids)

            raw_meta = dict(payload.get("snapshot_meta", {}))
            label = (
                raw_meta.get("label")
                or raw_meta.get("snapshot_id")
                or raw_meta.get("reason")
                or path.stem
            )
            frame_meta = {
                "index": index,
                "label": str(label),
                "timestamp": raw_meta.get("timestamp") or raw_meta.get("exported_at") or "",
                "runtime_step": raw_meta.get("runtime_step"),
                "world_t": raw_meta.get("world_t"),
                "reason": raw_meta.get("reason"),
                "trigger_mode": raw_meta.get("trigger_mode"),
                "signal_id": raw_meta.get("signal_id"),
                "domain_id": raw_meta.get("domain_id"),
                "path": str(path),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "added_node_ids": added_node_ids,
                "removed_node_ids": removed_node_ids,
                "added_edge_ids": added_edge_ids,
                "removed_edge_ids": removed_edge_ids,
                "added_node_count": len(added_node_ids),
                "removed_node_count": len(removed_node_ids),
                "added_edge_count": len(added_edge_ids),
                "removed_edge_count": len(removed_edge_ids),
            }
            graph_data = {
                "nodes": nodes,
                "links": [
                    {
                        **edge,
                        "source": edge["source"],
                        "target": edge["target"],
                    }
                    for edge in edges
                ],
            }
            frames.append(SnapshotFrame(path=path, meta=frame_meta, graph_data=graph_data))
            previous_node_ids = node_ids
            previous_edge_ids = edge_ids
        return frames

    def _render_html(self, payload: dict[str, Any]) -> str:
        data_json = json.dumps(payload, ensure_ascii=False)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Freeman Graph Evolution</title>
  <style>
    :root {{
      --bg-0: #08111f;
      --bg-1: #0f1d32;
      --panel: rgba(9, 18, 31, 0.84);
      --panel-border: rgba(140, 181, 255, 0.18);
      --text: #ecf3ff;
      --muted: #97a8c6;
      --accent: #79d2ff;
      --accent-2: #7af0c4;
      --good: #7af0c4;
      --bad: #ff7e87;
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
    .left {{
      top: 20px;
      left: 20px;
      width: 360px;
      padding: 18px;
    }}
    .right {{
      top: 20px;
      right: 20px;
      width: 360px;
      max-height: calc(100% - 40px);
      padding: 18px;
      overflow: auto;
    }}
    .bottom {{
      left: 20px;
      bottom: 20px;
      width: min(900px, calc(100% - 40px));
      padding: 14px 16px;
    }}
    h1, h2 {{
      margin: 0 0 10px;
    }}
    h1 {{ font-size: 20px; }}
    h2 {{ font-size: 15px; color: var(--accent); }}
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
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .card strong {{
      display: block;
      font-size: 20px;
      color: var(--text);
      margin-bottom: 4px;
    }}
    .field {{
      display: grid;
      gap: 6px;
      margin-bottom: 10px;
    }}
    .field label {{
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .field input,
    .field select,
    .field button {{
      width: 100%;
      background: rgba(255,255,255,0.06);
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 12px;
      padding: 10px 12px;
      outline: none;
    }}
    .buttons {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }}
    .timeline-meta {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .badge {{
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
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
    .stack {{
      pointer-events: none;
    }}
    .stack > .panel {{
      pointer-events: auto;
    }}
    @media (max-width: 920px) {{
      body {{ overflow: auto; }}
      #scene {{ position: relative; min-height: 72vh; }}
      .left, .right, .bottom {{
        position: static;
        width: auto;
        max-height: none;
        margin: 20px;
      }}
      .stack {{
        display: grid;
        gap: 0;
      }}
    }}
  </style>
</head>
<body>
  <div id="scene"></div>
  <div class="stack">
    <section class="panel left">
      <h1>Freeman Graph Evolution</h1>
      <p>Timeline viewer for knowledge-graph snapshots. Added nodes are highlighted in mint; stable nodes keep their semantic bucket color.</p>
      <div class="stats">
        <div class="card"><strong id="stat-nodes">0</strong><span class="small">Nodes</span></div>
        <div class="card"><strong id="stat-links">0</strong><span class="small">Links</span></div>
        <div class="card"><strong id="stat-snapshot">0 / 0</strong><span class="small">Frame</span></div>
      </div>
      <div class="field">
        <label for="search">Search</label>
        <input id="search" type="text" placeholder="label, bucket, content">
      </div>
      <div class="field">
        <label for="bucket">Bucket</label>
        <select id="bucket"></select>
      </div>
      <div class="field">
        <label for="confidence">Minimum Confidence</label>
        <input id="confidence" type="range" min="0" max="1" step="0.05" value="0">
        <span class="small" id="confidence-value">0.00</span>
      </div>
      <div class="field">
        <label for="timeline">Timeline</label>
        <input id="timeline" type="range" min="0" max="0" step="1" value="0">
      </div>
      <div class="buttons">
        <button id="prev">Prev</button>
        <button id="play">Play</button>
        <button id="next">Next</button>
      </div>
      <div class="timeline-meta" id="timeline-meta"></div>
    </section>

    <aside class="panel right">
      <h2>Selection</h2>
      <p id="selection-blurb">Click a node to inspect its local neighborhood in the selected snapshot.</p>
      <div id="selection"></div>
      <div style="margin-top:14px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.08);">
        <h2>Snapshot Delta</h2>
        <div id="delta"></div>
      </div>
    </aside>

    <section class="panel bottom">
      <div class="small"><strong style="color: var(--text);">Navigation:</strong> drag to orbit, scroll to zoom, right-drag to pan.</div>
      <div class="badge-row" id="badges"></div>
    </section>
  </div>

  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
  <script src="https://unpkg.com/3d-force-graph@1.76.0/dist/3d-force-graph.min.js"></script>
  <script src="https://unpkg.com/three-spritetext@1.9.6/dist/three-spritetext.min.js"></script>
  <script>
    const payload = {data_json};
    const snapshots = payload.snapshots || [];
    const sceneEl = document.getElementById('scene');
    const statNodesEl = document.getElementById('stat-nodes');
    const statLinksEl = document.getElementById('stat-links');
    const statSnapshotEl = document.getElementById('stat-snapshot');
    const selectionBlurbEl = document.getElementById('selection-blurb');
    const selectionEl = document.getElementById('selection');
    const deltaEl = document.getElementById('delta');
    const timelineMetaEl = document.getElementById('timeline-meta');
    const badgesEl = document.getElementById('badges');
    const searchEl = document.getElementById('search');
    const bucketEl = document.getElementById('bucket');
    const confidenceEl = document.getElementById('confidence');
    const confidenceValueEl = document.getElementById('confidence-value');
    const timelineEl = document.getElementById('timeline');
    const prevEl = document.getElementById('prev');
    const playEl = document.getElementById('play');
    const nextEl = document.getElementById('next');

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

    const state = {{
      snapshotIndex: Math.max(0, snapshots.length - 1),
      selectedNodeId: null,
      hoveredNodeId: null,
      playing: false,
      timerId: null
    }};

    const Graph = ForceGraph3D()(sceneEl)
      .backgroundColor('rgba(0,0,0,0)')
      .showNavInfo(false)
      .nodeRelSize(5)
      .cooldownTicks(160)
      .d3AlphaDecay(0.024)
      .d3VelocityDecay(0.24)
      .linkOpacity(0.18)
      .nodeLabel(node => `${{node.label}}\\n${{node.node_type}} | bucket=${{node.metadata?.bucket || 'n/a'}} | confidence=${{Number(node.confidence || 0).toFixed(2)}}`);
    Graph.width(sceneEl.clientWidth || window.innerWidth);
    Graph.height(sceneEl.clientHeight || window.innerHeight);
    sceneEl.tabIndex = 0;
    const controls = typeof Graph.controls === 'function' ? Graph.controls() : null;
    if (controls) {{
      controls.enabled = true;
      if ('zoomSpeed' in controls) controls.zoomSpeed = 5.2;
      if ('rotateSpeed' in controls) controls.rotateSpeed = 1.35;
      if ('panSpeed' in controls) controls.panSpeed = 1.2;
      if ('noRotate' in controls) controls.noRotate = false;
      if ('noZoom' in controls) controls.noZoom = false;
      if ('noPan' in controls) controls.noPan = false;
      if ('staticMoving' in controls) controls.staticMoving = true;
      if ('dynamicDampingFactor' in controls) controls.dynamicDampingFactor = 0.12;
    }}
    const chargeForce = Graph.d3Force('charge');
    if (chargeForce) chargeForce.strength(-180);
    const linkForce = Graph.d3Force('link');
    if (linkForce) linkForce.distance(link => 60 + Math.min(70, ((link.weight || 1) * 10)));
    const threeScene = Graph.scene();
    threeScene.add(new THREE.AmbientLight(0xb9d8ff, 1.35));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
    keyLight.position.set(120, 160, 180);
    threeScene.add(keyLight);
    const rimLight = new THREE.PointLight(0x7af0c4, 1.6, 800);
    rimLight.position.set(-180, -90, 160);
    threeScene.add(rimLight);

    function activateSceneInteraction() {{
      const active = document.activeElement;
      if (active && ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(active.tagName)) active.blur();
      sceneEl.focus({{ preventScroll: true }});
      if (controls && typeof controls.update === 'function') controls.update();
    }}

    function currentSnapshot() {{
      return snapshots[state.snapshotIndex];
    }}

    function graphOfCurrentSnapshot() {{
      return currentSnapshot().graphData;
    }}

    function snapshotMeta() {{
      return currentSnapshot().meta;
    }}

    function nodeById() {{
      return new Map(graphOfCurrentSnapshot().nodes.map(node => [node.id, node]));
    }}

    function outgoingById() {{
      const mapped = new Map();
      graphOfCurrentSnapshot().links.forEach(link => {{
        const source = typeof link.source === 'object' ? link.source.id : link.source;
        if (!mapped.has(source)) mapped.set(source, []);
        mapped.get(source).push(link);
      }});
      return mapped;
    }}

    function incomingById() {{
      const mapped = new Map();
      graphOfCurrentSnapshot().links.forEach(link => {{
        const target = typeof link.target === 'object' ? link.target.id : link.target;
        if (!mapped.has(target)) mapped.set(target, []);
        mapped.get(target).push(link);
      }});
      return mapped;
    }}

    function populateBuckets() {{
      const values = Array.from(new Set(graphOfCurrentSnapshot().nodes.map(node => node.metadata?.bucket).filter(Boolean))).sort();
      const current = bucketEl.value;
      bucketEl.innerHTML = '';
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'All buckets';
      bucketEl.appendChild(option);
      values.forEach(value => {{
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = value;
        bucketEl.appendChild(opt);
      }});
      if (values.includes(current)) bucketEl.value = current;
    }}

    function renderMeta() {{
      const meta = snapshotMeta();
      statNodesEl.textContent = String(meta.node_count);
      statLinksEl.textContent = String(meta.edge_count);
      statSnapshotEl.textContent = `${{meta.index + 1}} / ${{snapshots.length}}`;
      timelineMetaEl.innerHTML = `
        <div class="small"><strong style="color:var(--text)">Snapshot:</strong> ${{meta.label}}</div>
        <div class="small"><strong style="color:var(--text)">Timestamp:</strong> ${{meta.timestamp || 'n/a'}}</div>
        <div class="small"><strong style="color:var(--text)">Reason:</strong> ${{meta.reason || 'n/a'}} | <strong style="color:var(--text)">runtime_step:</strong> ${{meta.runtime_step ?? 'n/a'}}</div>
      `;
      deltaEl.innerHTML = `
        <div class="badge-row">
          <span class="badge good">+ nodes: ${{meta.added_node_count}}</span>
          <span class="badge bad">- nodes: ${{meta.removed_node_count}}</span>
          <span class="badge good">+ edges: ${{meta.added_edge_count}}</span>
          <span class="badge bad">- edges: ${{meta.removed_edge_count}}</span>
        </div>
        <ul class="list">
          <li>Snapshot path: ${{meta.path}}</li>
          <li>Signal id: ${{meta.signal_id || 'n/a'}}</li>
          <li>Trigger mode: ${{meta.trigger_mode || 'n/a'}}</li>
        </ul>
      `;
      badgesEl.innerHTML = `
        <span class="badge">source: ${{payload.meta.source}}</span>
        <span class="badge">exported: ${{payload.meta.exported_at}}</span>
        <span class="badge good">added nodes highlighted</span>
      `;
    }}

    function colorForNode(node) {{
      if ((snapshotMeta().added_node_ids || []).includes(node.id)) return '#7af0c4';
      return bucketPalette[node.metadata?.bucket] || '#9fb3d1';
    }}

    function filteredGraph() {{
      const search = searchEl.value.trim().toLowerCase();
      const bucket = bucketEl.value;
      const minConfidence = Number(confidenceEl.value);
      const baseGraph = graphOfCurrentSnapshot();
      const visibleNodeIds = new Set(baseGraph.nodes.filter(node => {{
        const haystack = [
          node.id,
          node.label,
          node.content,
          node.node_type,
          node.status,
          node.metadata?.bucket,
          ...(node.sources || []),
        ].join(' ').toLowerCase();
        if (search && !haystack.includes(search)) return false;
        if (bucket && node.metadata?.bucket !== bucket) return false;
        if ((node.confidence || 0) < minConfidence) return false;
        return true;
      }}).map(node => node.id));
      const links = baseGraph.links.filter(link => {{
        const source = typeof link.source === 'object' ? link.source.id : link.source;
        const target = typeof link.target === 'object' ? link.target.id : link.target;
        return visibleNodeIds.has(source) && visibleNodeIds.has(target);
      }});
      return {{
        nodes: baseGraph.nodes.filter(node => visibleNodeIds.has(node.id)).map(node => ({{ ...node }})),
        links: links.map(link => ({{ ...link }}))
      }};
    }}

    function decorateNode(node) {{
      const radius = 3.2 + Math.min(7.5, Number(node.metadata?.degree || 0) * 0.18) + Number(node.confidence || 0) * 3.5;
      const isAdded = (snapshotMeta().added_node_ids || []).includes(node.id);
      const material = new THREE.MeshStandardMaterial({{
        color: colorForNode(node),
        emissive: isAdded ? 0x7af0c4 : (state.selectedNodeId === node.id ? 0xffffff : 0x26364f),
        emissiveIntensity: isAdded ? 0.42 : state.selectedNodeId === node.id ? 0.35 : 0.12,
        roughness: 0.28,
        metalness: 0.22,
        transparent: true,
        opacity: 0.96
      }});
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(radius, 18, 18), material);
      const group = new THREE.Group();
      group.add(sphere);
      if (isAdded || state.selectedNodeId === node.id || Number(node.metadata?.degree || 0) >= 10) {{
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

    function renderSelection(node) {{
      if (!node) {{
        selectionBlurbEl.textContent = 'Click a node to inspect its local neighborhood in the selected snapshot.';
        selectionEl.innerHTML = '';
        return;
      }}
      const nodeMap = nodeById();
      const outgoing = (outgoingById().get(node.id) || []).slice(0, 8);
      const incoming = (incomingById().get(node.id) || []).slice(0, 8);
      selectionBlurbEl.textContent = node.content || 'No content.';
      selectionEl.innerHTML = `
        <ul class="list">
          <li><strong>label:</strong> ${{node.label}}</li>
          <li><strong>id:</strong> ${{node.id}}</li>
          <li><strong>type:</strong> ${{node.node_type}}</li>
          <li><strong>bucket:</strong> ${{node.metadata?.bucket || 'n/a'}}</li>
          <li><strong>confidence:</strong> ${{Number(node.confidence || 0).toFixed(2)}}</li>
        </ul>
        <div style="margin-top:12px;">
          <h2>Outgoing</h2>
          ${{outgoing.length ? `<ul class="list">${{outgoing.map(link => {{
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return `<li><strong>${{link.relation_type}}</strong> → ${{nodeMap.get(targetId)?.label || targetId}}</li>`;
          }}).join('')}}</ul>` : '<p>No outgoing links.</p>'}}
        </div>
        <div style="margin-top:12px;">
          <h2>Incoming</h2>
          ${{incoming.length ? `<ul class="list">${{incoming.map(link => {{
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            return `<li>${{nodeMap.get(sourceId)?.label || sourceId}} → <strong>${{link.relation_type}}</strong></li>`;
          }}).join('')}}</ul>` : '<p>No incoming links.</p>'}}
        </div>
      `;
    }}

    function refreshGraph() {{
      const graphData = filteredGraph();
      Graph
        .graphData(graphData)
        .nodeThreeObject(node => decorateNode(node))
        .nodeThreeObjectExtend(false)
        .linkColor(link => 'rgba(160,188,255,0.32)')
        .linkDirectionalParticles(link => 0);
      renderMeta();
    }}

    function setSnapshotIndex(index) {{
      state.snapshotIndex = Math.max(0, Math.min(snapshots.length - 1, index));
      timelineEl.value = String(state.snapshotIndex);
      state.selectedNodeId = null;
      renderSelection(null);
      populateBuckets();
      refreshGraph();
      activateSceneInteraction();
    }}

    function togglePlayback() {{
      state.playing = !state.playing;
      playEl.textContent = state.playing ? 'Pause' : 'Play';
      if (!state.playing) {{
        window.clearInterval(state.timerId);
        state.timerId = null;
        return;
      }}
      state.timerId = window.setInterval(() => {{
        if (state.snapshotIndex >= snapshots.length - 1) {{
          state.playing = false;
          playEl.textContent = 'Play';
          window.clearInterval(state.timerId);
          state.timerId = null;
          return;
        }}
        setSnapshotIndex(state.snapshotIndex + 1);
      }}, 1400);
    }}

    confidenceEl.addEventListener('input', () => {{
      confidenceValueEl.textContent = Number(confidenceEl.value).toFixed(2);
      refreshGraph();
    }});
    searchEl.addEventListener('input', refreshGraph);
    bucketEl.addEventListener('input', refreshGraph);
    timelineEl.addEventListener('input', () => setSnapshotIndex(Number(timelineEl.value)));
    prevEl.addEventListener('click', () => setSnapshotIndex(state.snapshotIndex - 1));
    nextEl.addEventListener('click', () => setSnapshotIndex(state.snapshotIndex + 1));
    playEl.addEventListener('click', togglePlayback);

    Graph.onNodeClick(node => {{
      state.selectedNodeId = node.id;
      renderSelection(nodeById().get(node.id));
      activateSceneInteraction();
      const distance = 110;
      const distRatio = 1 + distance / Math.hypot(node.x || 1, node.y || 1, node.z || 1);
      Graph.cameraPosition(
        {{ x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio }},
        node,
        700
      );
      window.setTimeout(activateSceneInteraction, 760);
      Graph.refresh();
    }});
    Graph.onNodeHover(node => {{
      sceneEl.style.cursor = node ? 'pointer' : 'grab';
    }});
    Graph.onBackgroundClick(() => {{
      state.selectedNodeId = null;
      renderSelection(null);
      activateSceneInteraction();
      Graph.refresh();
    }});

    timelineEl.max = String(Math.max(0, snapshots.length - 1));
    Graph.cameraPosition({{ x: 0, y: 0, z: 340 }});
    confidenceValueEl.textContent = Number(confidenceEl.value).toFixed(2);
    setSnapshotIndex(state.snapshotIndex);
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


__all__ = ["KnowledgeGraphEvolutionExporter"]
