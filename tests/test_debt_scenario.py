"""Integration test for a GIM-style Japan debt crisis scenario."""

from __future__ import annotations

import json

from freeman.domain.compiler import DomainCompiler
from freeman.game.runner import GameRunner, SimConfig


def _japan_debt_schema() -> dict:
    return {
        "domain_id": "japan_debt",
        "name": "Japan Debt Stress Scenario",
        "actors": [
            {"id": "japan", "name": "Japan", "state": {"influence": 0.72}},
            {"id": "usa", "name": "USA", "state": {"influence": 0.92}},
            {"id": "china", "name": "China", "state": {"influence": 0.88}},
            {"id": "eu", "name": "EU", "state": {"influence": 0.84}},
        ],
        "resources": [
            {
                "id": "japan_debt_ratio",
                "name": "Japan Debt Ratio",
                "value": 2.6,
                "unit": "ratio",
                "min_value": 0.0,
                "max_value": 5.0,
                "evolution_type": "stock_flow",
                "evolution_params": {"delta": 0.02, "phi_params": {"base_inflow": 0.08}},
                "conserved": False,
            },
            {
                "id": "japan_gdp_growth",
                "name": "Japan GDP Growth",
                "value": 0.01,
                "unit": "rate",
                "min_value": -1.0,
                "max_value": 1.0,
                "evolution_type": "linear",
                "evolution_params": {
                    "a": 0.9,
                    "c": 0.005,
                    "coupling_weights": {"japan_debt_ratio": -0.02},
                },
            },
            {
                "id": "japan_political_stability",
                "name": "Japan Political Stability",
                "value": 0.65,
                "unit": "index",
                "min_value": 0.0,
                "max_value": 1.0,
                "evolution_type": "linear",
                "evolution_params": {
                    "a": 0.97,
                    "c": 0.03,
                    "coupling_weights": {"japan_gdp_growth": 0.1},
                },
            },
            {
                "id": "global_risk_appetite",
                "name": "Global Risk Appetite",
                "value": 0.5,
                "unit": "index",
                "min_value": 0.0,
                "max_value": 1.0,
                "evolution_type": "logistic",
                "evolution_params": {
                    "r": 0.05,
                    "K": 1.0,
                    "coupling_weights": {
                        "japan_political_stability": 0.03,
                        "japan_debt_ratio": -0.02,
                    },
                },
            },
        ],
        "relations": [],
        "outcomes": [
            {
                "id": "debt_crisis",
                "label": "Debt Crisis",
                "scoring_weights": {
                    "japan_debt_ratio": 2.0,
                    "japan_gdp_growth": -1.5,
                    "japan_political_stability": -1.0,
                },
            },
            {
                "id": "stable",
                "label": "Stable",
                "scoring_weights": {
                    "japan_debt_ratio": -1.0,
                    "japan_gdp_growth": 1.0,
                    "japan_political_stability": 1.0,
                },
            },
        ],
        "causal_dag": [
            {"source": "japan_debt_ratio", "target": "japan_gdp_growth", "expected_sign": "-", "strength": "strong"},
            {
                "source": "japan_gdp_growth",
                "target": "japan_political_stability",
                "expected_sign": "+",
                "strength": "strong",
            },
            {
                "source": "japan_political_stability",
                "target": "global_risk_appetite",
                "expected_sign": "+",
                "strength": "weak",
            },
            {
                "source": "japan_debt_ratio",
                "target": "global_risk_appetite",
                "expected_sign": "-",
                "strength": "weak",
            },
        ],
        "metadata": {"base_year": 2025, "time_unit": "year"},
    }


def test_japan_debt_scenario_runs_end_to_end() -> None:
    world = DomainCompiler().compile(_japan_debt_schema())
    result = GameRunner(SimConfig(max_steps=30, level2_check_every=5)).run(world, policies=[])

    hard = [violation for violation in result.violations if violation.severity == "hard"]
    assert hard == []
    assert result.steps_run == 30
    assert result.trajectory[-1]["resources"]["japan_debt_ratio"]["value"] > result.trajectory[0]["resources"]["japan_debt_ratio"]["value"]
    assert result.outcome_probs[-1]["debt_crisis"] > result.outcome_probs[0]["debt_crisis"]
    assert result.confidence > 0.0

    parsed = json.loads(result.to_json())
    assert parsed["domain_id"] == "japan_debt"
