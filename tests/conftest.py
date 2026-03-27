"""Shared pytest fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from freeman.domain.compiler import DomainCompiler


def build_water_market_schema(domain_id: str = "water_market") -> dict:
    """Return a stable test schema for the water-market domain."""

    return {
        "domain_id": domain_id,
        "name": "Water Market Simulation",
        "description": "Regional water allocation and conflict dynamics.",
        "actors": [
            {
                "id": "country_a",
                "name": "Country A",
                "state": {"influence": 0.6},
                "metadata": {"population_millions": 45},
            },
            {
                "id": "country_b",
                "name": "Country B",
                "state": {"influence": 0.55},
                "metadata": {"population_millions": 52},
            },
        ],
        "resources": [
            {
                "id": "water_stock",
                "name": "Water Stock",
                "value": 1000.0,
                "unit": "km3",
                "min_value": 0.0,
                "max_value": 2000.0,
                "conserved": True,
                "evolution_type": "stock_flow",
                "evolution_params": {
                    "delta": 0.03,
                    "phi_params": {
                        "base_inflow": 30.0,
                        "coupling_weights": {
                            "conflict_level": -0.5
                        }
                    }
                }
            },
            {
                "id": "agriculture_output",
                "name": "Agriculture Output",
                "value": 300.0,
                "unit": "index",
                "min_value": 0.0,
                "max_value": 800.0,
                "evolution_type": "linear",
                "evolution_params": {
                    "a": 0.7,
                    "b": 0.0,
                    "c": 10.0,
                    "coupling_weights": {
                        "water_stock": 0.06
                    }
                }
            },
            {
                "id": "conflict_level",
                "name": "Conflict Level",
                "value": 10.0,
                "unit": "index",
                "min_value": 0.0,
                "max_value": 100.0,
                "evolution_type": "linear",
                "evolution_params": {
                    "a": 0.6,
                    "b": 0.0,
                    "c": 15.0,
                    "coupling_weights": {
                        "water_stock": -0.0015,
                        "agriculture_output": -0.003
                    }
                }
            }
        ],
        "relations": [
            {
                "source_id": "country_a",
                "target_id": "country_b",
                "relation_type": "water_dependency",
                "weights": {"dependency": 0.4},
            }
        ],
        "outcomes": [
            {
                "id": "cooperation",
                "label": "Cooperation",
                "scoring_weights": {
                    "water_stock": 0.01,
                    "agriculture_output": 0.02,
                    "conflict_level": -0.08
                },
                "description": "High water availability and low conflict"
            },
            {
                "id": "water_crisis",
                "label": "Water Crisis",
                "scoring_weights": {
                    "water_stock": -0.03,
                    "conflict_level": 0.06
                },
                "description": "Water depletion under regional stress"
            },
            {
                "id": "conflict_escalation",
                "label": "Conflict Escalation",
                "scoring_weights": {
                    "conflict_level": 0.08,
                    "agriculture_output": -0.02
                },
                "description": "Escalating conflict around water access"
            }
        ],
        "causal_dag": [
            {
                "source": "water_stock",
                "target": "agriculture_output",
                "expected_sign": "+",
                "strength": "strong"
            },
            {
                "source": "conflict_level",
                "target": "water_stock",
                "expected_sign": "-",
                "strength": "strong"
            },
            {
                "source": "water_stock",
                "target": "conflict_level",
                "expected_sign": "-",
                "strength": "strong"
            }
        ],
        "exogenous_inflows": {
            "water_stock": 30.0
        },
        "metadata": {
            "base_year": 2025,
            "time_unit": "year"
        }
    }


@pytest.fixture
def water_market_schema() -> dict:
    """Return the default water-market schema."""

    return copy.deepcopy(build_water_market_schema())


@pytest.fixture
def water_market_world(water_market_schema: dict):
    """Compile and return the default water-market world."""

    return DomainCompiler().compile(water_market_schema)


@pytest.fixture
def gim15_schema() -> dict:
    """Load the bundled GIM15-compatible schema."""

    profile_path = Path(__file__).resolve().parents[1] / "freeman" / "domain" / "profiles" / "gim15.json"
    return json.loads(profile_path.read_text(encoding="utf-8"))
