"""Tests for human overrides and simulation diffs."""

from __future__ import annotations

from freeman.core.types import CausalEdge, Outcome, Resource
from freeman.core.world import WorldState
from freeman.interface.api import InterfaceAPI
from freeman.interface.modeloverride import ModelOverrideAPI


def _world() -> WorldState:
    return WorldState(
        domain_id="override_demo",
        t=0,
        actors={},
        resources={
            "x": Resource(id="x", name="X", value=10.0, unit="u", evolution_type="linear", evolution_params={"a": 0.9}),
            "y": Resource(
                id="y",
                name="Y",
                value=5.0,
                unit="u",
                evolution_type="linear",
                evolution_params={"a": 0.8, "coupling_weights": {"x": 0.3}},
            ),
        },
        relations=[],
        outcomes={
            "good": Outcome(id="good", label="Good", scoring_weights={"x": 0.1, "y": 0.1}),
            "bad": Outcome(id="bad", label="Bad", scoring_weights={"x": -0.1, "y": -0.1}),
        },
        causal_dag=[CausalEdge(source="x", target="y", expected_sign="-", strength="strong")],
    )


def test_model_override_api_preserves_machine_world_and_tracks_audit_log() -> None:
    api = ModelOverrideAPI()
    world = _world()
    api.register_domain(world.domain_id, world)

    api.patch_params(world.domain_id, {"resources.y.evolution_params.a": 0.4})
    api.patch_edge(world.domain_id, "x->y", "+")
    rerun = api.rerun_domain(world.domain_id)
    diff = api.get_diff(world.domain_id)

    record = api.records[world.domain_id]

    assert record.machine_world.resources["y"].evolution_params["a"] == 0.8
    assert record.current_world.resources["y"].evolution_params["a"] == 0.4
    assert record.current_world.causal_dag[0].expected_sign == "+"
    assert len(record.audit_log) == 2
    assert rerun["domain_id"] == world.domain_id
    assert len(diff["override_history"]) == 2
    assert any(change["path"] == "causal_dag.0.expected_sign" for change in diff["changes"])


def test_interface_api_exposes_override_routes() -> None:
    world = _world()
    api = InterfaceAPI()
    api.register_domain(world)

    patch_result = api.patch_domain_params(world.domain_id, {"resources.x.value": 12.0})
    edge_result = api.patch_domain_edge(world.domain_id, "x->y", "+")
    rerun = api.rerun_domain(world.domain_id)
    diff = api.get_domain_diff(world.domain_id)

    assert patch_result["domain_id"] == world.domain_id
    assert edge_result["version"] == 2
    assert rerun["domain_id"] == world.domain_id
    assert len(diff["override_history"]) == 2
