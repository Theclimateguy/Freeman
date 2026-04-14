from __future__ import annotations

import json

from scripts.seed_climate_kg import build_seed_graph, write_seed_graph


def test_climate_seed_graph_has_expected_shape() -> None:
    payload = build_seed_graph()

    assert payload["backend"] == "networkx-json"
    assert payload["seed_version"] == "climate_seed_v1"
    assert len(payload["nodes"]) >= 200
    assert len(payload["edges"]) >= 400

    node_ids = {node["id"] for node in payload["nodes"]}
    assert len(node_ids) == len(payload["nodes"])
    assert {
        "climate:greenhouse_gases",
        "climate:global_warming",
        "climate:sea_level_rise",
        "climate:physical_risk",
        "climate:economic_loss",
        "climate:financial_stability",
        "climate:mitigation_policy",
        "climate:carbon_pricing",
        "climate:tcfd_framework",
        "climate:climate_disclosure",
    }.issubset(node_ids)

    for edge in payload["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids

    edge_triples = {(edge["source"], edge["relation_type"], edge["target"]) for edge in payload["edges"]}
    assert ("climate:greenhouse_gases", "type_of", "climate:forcing_driver") in edge_triples
    assert ("climate:carbon_pricing", "type_of", "climate:mitigation_policy") in edge_triples
    assert ("climate:tcfd_framework", "governs", "climate:climate_disclosure") in edge_triples
    assert ("climate:global_warming", "causes", "climate:sea_level_rise") in edge_triples


def test_write_seed_graph_emits_seed_and_memory_payloads(tmp_path) -> None:
    seed_path = tmp_path / "kg_climate_seed.json"
    memory_path = tmp_path / "kg_climate.json"
    memory_path.write_text('{"backend":"networkx-json","nodes":[],"edges":[]}', encoding="utf-8")

    written_seed, written_memory = write_seed_graph(seed_path=seed_path, memory_path=memory_path)

    assert written_seed == seed_path
    assert written_memory == memory_path
    assert seed_path.exists()
    assert memory_path.exists()
    assert memory_path.with_suffix(".json.preseed.bak").exists()

    seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
    memory_payload = json.loads(memory_path.read_text(encoding="utf-8"))
    assert seed_payload["seed_version"] == memory_payload["seed_version"] == "climate_seed_v1"
    assert memory_payload["json_path"] == str(memory_path)
