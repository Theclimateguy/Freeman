"""Tests for the automated LLM repair loop."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from freeman.core.transition import step_world
from freeman.domain.compiler import DomainCompiler
from freeman.exceptions import HardStopException
from freeman.game.runner import SimConfig
from freeman.llm.orchestrator import DeepSeekFreemanOrchestrator
from freeman.verifier.level1 import level1_check
from freeman.verifier.level2 import level2_check


class RepairingStubClient:
    """Deterministic stand-in for DeepSeek during repair-loop tests."""

    def __init__(self, initial_package: Dict[str, Any]) -> None:
        self.initial_package = copy.deepcopy(initial_package)
        self.feedback_log: List[List[Dict[str, Any]]] = []
        self.model = "repairing-stub"

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Return the initial malformed package on the first synthesis turn."""

        return copy.deepcopy(self.initial_package)

    def repair_schema(
        self,
        package: Dict[str, Any],
        violations: List[Dict[str, Any]],
        *,
        domain_description: str = "",
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """Repair the package deterministically from structured verifier feedback."""

        self.feedback_log.append(copy.deepcopy(violations))
        repaired = copy.deepcopy(package)
        resources = {resource["id"]: resource for resource in repaired["schema"]["resources"]}
        check_names = {violation["check_name"] for violation in violations}

        if {"spectral_radius", "null_action_convergence", "shock_decay"} & check_names:
            for resource in resources.values():
                params = resource.get("evolution_params", {})
                if resource["evolution_type"] == "linear":
                    params["a"] = min(float(params.get("a", 0.9)), 0.7)
                    couplings = params.setdefault("coupling_weights", {})
                    for key, value in list(couplings.items()):
                        couplings[key] = max(min(float(value), 0.01), -0.01)
                if resource["evolution_type"] == "stock_flow":
                    params["delta"] = max(float(params.get("delta", 0.05)), 0.05)
                    couplings = params.setdefault("phi_params", {}).setdefault("coupling_weights", {})
                    for key, value in list(couplings.items()):
                        couplings[key] = max(min(float(value), 0.5), -0.5)

        if "sign_consistency" in check_names:
            for violation in violations:
                edge = violation["details"]["edge"]
                target = edge["target"]
                source = edge["source"]
                expected_sign = 1.0 if edge["expected_sign"] == "+" else -1.0
                resource = resources.get(target)
                if resource is None:
                    continue
                params = resource.get("evolution_params", {})
                if source in params.get("coupling_weights", {}):
                    params["coupling_weights"][source] = abs(float(params["coupling_weights"][source])) * expected_sign
                phi_weights = params.get("phi_params", {}).get("coupling_weights", {})
                if source in phi_weights:
                    phi_weights[source] = abs(float(phi_weights[source])) * expected_sign

        return repaired


def test_repair_loop_convergence(water_market_schema: Dict[str, Any]) -> None:
    """The orchestrator should repair malformed parameters into a verifier-clean schema."""

    malformed_schema = copy.deepcopy(water_market_schema)
    for resource in malformed_schema["resources"]:
        if resource["id"] == "water_stock":
            resource["evolution_params"]["phi_params"]["coupling_weights"]["conflict_level"] = -5.0
        if resource["id"] == "agriculture_output":
            resource["evolution_params"]["a"] = 1.25
            resource["evolution_params"]["coupling_weights"]["water_stock"] = -0.2
        if resource["id"] == "conflict_level":
            resource["evolution_params"]["a"] = 1.1

    client = RepairingStubClient(
        {
            "schema": malformed_schema,
            "policies": [],
            "assumptions": ["Initial package intentionally contains unstable coefficients for repair-loop testing."],
        }
    )
    orchestrator = DeepSeekFreemanOrchestrator(client)

    package, world_id, attempts, repair_history = orchestrator.compile_and_repair(
        "Repair a malformed water-market package.",
        max_retries=5,
        trial_steps=3,
    )

    world = DomainCompiler().compile(package["schema"])
    repair_config = SimConfig(convergence_check_steps=250, convergence_epsilon=3.0e-2)
    l1_violations = level1_check(world.clone(), repair_config)
    l2_violations = level2_check(world.clone(), world.causal_dag)
    l0_violations = []
    current = world.clone()
    try:
        for _ in range(3):
            current, step_violations = step_world(current, [])
            l0_violations.extend(step_violations)
    except HardStopException as exc:  # pragma: no cover - failure path for easier debugging
        l0_violations.extend(exc.violations)

    assert package["schema"]["domain_id"] == malformed_schema["domain_id"]
    assert world_id.startswith("water_market:")
    assert attempts == 3
    assert len(repair_history) == 2
    assert l1_violations == []
    assert l2_violations == []
    assert l0_violations == []

    first_feedback = client.feedback_log[0]
    spectral = next(violation for violation in first_feedback if violation["check_name"] == "spectral_radius")
    assert spectral["details"]["field"] == "jacobian.spectral_radius"
    assert spectral["details"]["observed"] != "hard_stop"
    assert spectral["details"]["expected_max"] == 1.0

    second_feedback = client.feedback_log[1]
    sign_violation = next(violation for violation in second_feedback if violation["check_name"] == "sign_consistency")
    assert sign_violation["details"]["repair_targets"] == [
        "resources.agriculture_output.evolution_params.coupling_weights.water_stock"
    ]
    assert sign_violation["details"]["observed"] == "-"
    assert sign_violation["details"]["expected"] == "+"
