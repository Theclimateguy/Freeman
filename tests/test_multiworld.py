"""Tests for shared-resource multi-domain simulation."""

from __future__ import annotations

from freeman.core.multiworld import MultiDomainWorld
from freeman.domain.compiler import DomainCompiler


def test_shared_resource_propagates_between_domains() -> None:
    compiler = DomainCompiler()
    domain_a = compiler.compile(
        {
            "domain_id": "river_basin_a",
            "actors": [],
            "resources": [
                {
                    "id": "shared_water",
                    "name": "Shared Water",
                    "value": 100.0,
                    "unit": "km3",
                    "evolution_type": "stock_flow",
                    "evolution_params": {
                        "delta": 0.0,
                        "phi_params": {"base_inflow": 10.0}
                    }
                }
            ],
            "relations": [],
            "outcomes": [
                {
                    "id": "ok",
                    "label": "OK",
                    "scoring_weights": {"shared_water": 1.0}
                },
                {
                    "id": "bad",
                    "label": "Bad",
                    "scoring_weights": {"shared_water": -1.0}
                }
            ],
            "causal_dag": [],
            "exogenous_inflows": {"shared_water": 10.0}
        }
    )
    domain_b = compiler.compile(
        {
            "domain_id": "river_basin_b",
            "actors": [],
            "resources": [
                {
                    "id": "shared_water",
                    "name": "Shared Water",
                    "value": 100.0,
                    "unit": "km3",
                    "evolution_type": "linear",
                    "evolution_params": {"a": 1.0, "c": 0.0}
                },
                {
                    "id": "downstream_output",
                    "name": "Downstream Output",
                    "value": 20.0,
                    "unit": "index",
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.5,
                        "c": 0.0,
                        "coupling_weights": {"shared_water": 0.05}
                    }
                }
            ],
            "relations": [],
            "outcomes": [
                {
                    "id": "ok",
                    "label": "OK",
                    "scoring_weights": {"downstream_output": 1.0}
                },
                {
                    "id": "bad",
                    "label": "Bad",
                    "scoring_weights": {"downstream_output": -1.0}
                }
            ],
            "causal_dag": [],
            "exogenous_inflows": {"shared_water": 0.0}
        }
    )

    multi = MultiDomainWorld([domain_a, domain_b], shared_resource_ids=["shared_water"])
    result = multi.step({})

    assert result.shared_state["shared_water"]["value"] == 110.0
    assert result.domains["river_basin_b"]["resources"]["shared_water"]["value"] == 110.0
    assert result.domains["river_basin_b"]["resources"]["downstream_output"]["value"] == 15.5
