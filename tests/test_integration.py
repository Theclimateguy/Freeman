"""End-to-end integration test for the v0.1/v0.2 baseline."""

from __future__ import annotations

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.game.runner import SimConfig
from freeman.interface.kgexport import KnowledgeGraphExporter
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.sessionlog import KGDelta, SessionLog
from freeman.verifier.verifier import Verifier


def test_end_to_end_pipeline_30_steps_reconciles_and_exports(tmp_path, water_market_schema) -> None:
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "knowledge_graph.json", auto_load=False, auto_save=True)
    session_log = SessionLog(session_id="integration")
    session_log.add_kg_delta(
        KGDelta(
            operation="add_node",
            payload={
                "node": KGNode(
                    id="signal:water",
                    label="Water stress alert",
                    content="Manual analyst alert on regional water stress.",
                    confidence=0.75,
                    metadata={"claim_key": "water_stress_alert"},
                ).snapshot()
            },
        )
    )

    sim_config = SimConfig(max_steps=30, convergence_check_steps=250, convergence_epsilon=3.0e-2, seed=7)
    pipeline = AnalysisPipeline(sim_config=sim_config, knowledge_graph=knowledge_graph)
    result = pipeline.run(water_market_schema, session_log=session_log)

    exporter = KnowledgeGraphExporter()
    html_path = exporter.export_html(knowledge_graph, tmp_path / "kg.html")
    html_3d_path = exporter.export_html_3d(knowledge_graph, tmp_path / "kg_3d.html")
    dot_path = exporter.export_dot(knowledge_graph, tmp_path / "kg.dot")
    jsonld_path = exporter.export_json_ld(knowledge_graph, tmp_path / "kg.jsonld")
    verification = Verifier(sim_config).run(result.world, levels=(1, 2))

    assert result.simulation["steps_run"] == 30
    assert result.reconciliation is not None
    assert result.reconciliation.processed_deltas >= 1
    assert knowledge_graph.json_path.exists()
    assert html_path.exists()
    assert html_3d_path.exists()
    assert dot_path.exists()
    assert jsonld_path.exists()
    assert "3d-force-graph" in html_3d_path.read_text(encoding="utf-8")
    assert not any(violation["severity"] == "hard" for violation in result.simulation["violations"])
    assert verification.passed is True
