"""Shared fixtures for Freeman lite tests."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_water_market_schema(domain_id: str = "water_market") -> dict:
    return {
        "domain_id": domain_id,
        "name": "Water Market Simulation",
        "description": "Regional water allocation and conflict dynamics.",
        "actors": [
            {"id": "country_a", "name": "Country A", "state": {"influence": 0.6}},
            {"id": "country_b", "name": "Country B", "state": {"influence": 0.55}},
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
                    "phi_params": {"base_inflow": 30.0, "coupling_weights": {"conflict_level": -0.5}},
                },
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
                    "coupling_weights": {"water_stock": 0.06},
                },
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
                    "coupling_weights": {"water_stock": -0.0015, "agriculture_output": -0.003},
                },
            },
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
                "scoring_weights": {"water_stock": 0.01, "agriculture_output": 0.02, "conflict_level": -0.08},
            },
            {
                "id": "water_crisis",
                "label": "Water Crisis",
                "scoring_weights": {"water_stock": -0.03, "conflict_level": 0.06},
            },
            {
                "id": "conflict_escalation",
                "label": "Conflict Escalation",
                "scoring_weights": {"conflict_level": 0.08, "agriculture_output": -0.02},
            },
        ],
        "causal_dag": [
            {"source": "water_stock", "target": "agriculture_output", "expected_sign": "+", "strength": "strong"},
            {"source": "conflict_level", "target": "water_stock", "expected_sign": "-", "strength": "strong"},
            {"source": "water_stock", "target": "conflict_level", "expected_sign": "-", "strength": "strong"},
        ],
        "exogenous_inflows": {"water_stock": 30.0},
        "metadata": {"base_year": 2025, "time_unit": "year"},
    }


@pytest.fixture
def water_market_schema() -> dict:
    return copy.deepcopy(build_water_market_schema())


@pytest.fixture
def lite_config_path(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "kg_state": str(tmp_path / "kg_state.json"),
                    "forecasts": str(tmp_path / "forecasts.json"),
                    "world_state": str(tmp_path / "world_state.json"),
                    "error_log": str(tmp_path / "errors.jsonl"),
                },
                "llm": {"provider": "none", "model": "", "base_url": "", "timeout_seconds": 30},
                "limits": {"max_llm_calls": 50, "max_simulation_steps": 5000},
                "signals": {"keywords": ["water", "drought"], "min_keyword_hits": 1, "conflict_threshold": 0.25},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def schema_path(tmp_path: Path, water_market_schema: dict) -> Path:
    path = tmp_path / "water_market.json"
    path.write_text(json.dumps(water_market_schema, indent=2, sort_keys=True), encoding="utf-8")
    return path
