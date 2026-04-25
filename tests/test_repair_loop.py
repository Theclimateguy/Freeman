"""Tests for the automated LLM repair loop."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

from freeman.core.transition import step_world
from freeman.domain.compiler import DomainCompiler
from freeman.exceptions import HardStopException
from freeman.game.runner import SimConfig
from freeman.llm.orchestrator import FreemanOrchestrator
from freeman.verifier.level1 import level1_check
from freeman.verifier.level2 import level2_check


class RepairingStubClient:
    """Deterministic stand-in for an LLM repair client during repair-loop tests."""

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
        """Return deterministic ETL-phase payloads from the initial package."""

        del temperature, max_tokens
        system = messages[0]["content"]
        package = copy.deepcopy(self.initial_package)
        schema = package.get("schema", {})
        if "domain skeletons" in system:
            skeleton_schema = {
                key: copy.deepcopy(schema[key])
                for key in ("domain_id", "name", "description", "actors", "resources", "outcomes")
                if key in schema
            }
            return {"schema": skeleton_schema, "assumptions": package.get("assumptions", [])}
        if "synthesize Freeman causal edges" in system:
            return {
                "causal_dag": copy.deepcopy(schema.get("causal_dag", [])),
                "actor_update_rules": copy.deepcopy(schema.get("actor_update_rules", {})),
                "policies": copy.deepcopy(package.get("policies", [])),
                "assumptions": [],
            }
        if "repair only Freeman causal edge signs" in system:
            payload = json.loads(messages[1]["content"])
            return {"causal_dag": copy.deepcopy(payload.get("causal_dag", schema.get("causal_dag", [])))}
        return package

    def repair_schema(
        self,
        package: Dict[str, Any],
        violations: List[Dict[str, Any]],
        *,
        domain_description: str = "",
        error_history: List[Dict[str, Any]] | None = None,
        verifier_schema_spec: Dict[str, Any] | None = None,
        repair_stage: str = "standard",
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """Repair the package deterministically from structured verifier feedback."""

        del domain_description, error_history, verifier_schema_spec, repair_stage, max_tokens
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
    orchestrator = FreemanOrchestrator(client)

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
    assert attempts == 2
    assert [record["etl_phase"] for record in repair_history[:2]] == ["skeleton", "edges"]
    assert [record["etl_phase"] for record in repair_history[2:]] == ["edges"]
    assert l1_violations == []
    assert l2_violations == []
    assert l0_violations == []

    first_feedback = client.feedback_log[0]
    spectral = next(violation for violation in first_feedback if violation["check_name"] == "spectral_radius")
    assert spectral["details"]["field"] == "jacobian.spectral_radius"

    repair_records = [record for record in repair_history if record.get("repair_stage")]
    assert repair_records[0]["repair_stage"] == "standard"
    assert repair_records[0]["verifier_error"]
    assert orchestrator.last_bootstrap_attempts == repair_history


def test_repair_sign_edges_updates_only_causal_couplings() -> None:
    """Surgical sign repair should avoid the full-package repair path."""

    malformed_schema = {
        "domain_id": "sign_repair_demo",
        "actors": [{"id": "planner", "name": "Planner", "state": {}}],
        "resources": [
            {
                "id": "input",
                "name": "Input",
                "value": 1.0,
                "unit": "index",
                "evolution_type": "linear",
                "evolution_params": {"a": 0.8, "b": 0.0, "c": 0.0, "coupling_weights": {}},
            },
            {
                "id": "output",
                "name": "Output",
                "value": 1.0,
                "unit": "index",
                "evolution_type": "linear",
                "evolution_params": {"a": 0.8, "b": 0.0, "c": 0.0, "coupling_weights": {"input": -0.2}},
            },
        ],
        "relations": [],
        "outcomes": [{"id": "score", "label": "Score", "scoring_weights": {"output": 1.0}}],
        "causal_dag": [{"source": "input", "target": "output", "expected_sign": "+", "strength": "strong"}],
    }
    package = {"schema": malformed_schema, "policies": [], "assumptions": []}
    world = DomainCompiler().compile(malformed_schema)
    violations = level2_check(world.clone(), world.causal_dag)
    assert any(violation.check_name == "sign_consistency" for violation in violations)

    client = RepairingStubClient(package)
    orchestrator = FreemanOrchestrator(client)
    repaired = orchestrator.repair_sign_edges(
        package,
        [{"phase": "level2", **violation.snapshot()} for violation in violations],
        domain_description="Repair only sign edges.",
    )

    repaired_world = DomainCompiler().compile(repaired["schema"])
    assert level2_check(repaired_world.clone(), repaired_world.causal_dag) == []
    assert client.feedback_log == []
    resources = {resource["id"]: resource for resource in repaired["schema"]["resources"]}
    assert resources["output"]["evolution_params"]["coupling_weights"]["input"] > 0.0


def test_repair_stage_escalates_with_attempt_history(water_market_schema: Dict[str, Any]) -> None:
    """Long repair runs should escalate prompt context deterministically."""

    malformed_schema = copy.deepcopy(water_market_schema)
    for resource in malformed_schema["resources"]:
        if resource["evolution_type"] == "linear":
            resource["evolution_params"]["a"] = 1.4
            couplings = resource["evolution_params"].setdefault("coupling_weights", {})
            for key in list(couplings):
                couplings[key] = 0.25

    class EscalatingStubClient(RepairingStubClient):
        def __init__(self, initial_package: Dict[str, Any], valid_schema: Dict[str, Any]) -> None:
            super().__init__(initial_package)
            self.valid_schema = copy.deepcopy(valid_schema)

        def repair_schema(
            self,
            package: Dict[str, Any],
            violations: List[Dict[str, Any]],
            *,
            domain_description: str = "",
            error_history: List[Dict[str, Any]] | None = None,
            verifier_schema_spec: Dict[str, Any] | None = None,
            repair_stage: str = "standard",
            max_tokens: int = 4000,
        ) -> Dict[str, Any]:
            del domain_description, verifier_schema_spec, max_tokens
            stage_payload = {
                "repair_stage": repair_stage,
                "history_len": len(error_history or []),
                "first_error": (error_history or [{}])[0].get("verifier_error"),
            }
            self.feedback_log.append([stage_payload])
            repaired = copy.deepcopy(package)
            if repair_stage == "standard":
                return repaired
            repaired["schema"] = copy.deepcopy(self.valid_schema)
            return repaired

    client = EscalatingStubClient(
        {
            "schema": malformed_schema,
            "policies": [],
            "assumptions": ["Escalation test."],
        },
        valid_schema=water_market_schema,
    )
    orchestrator = FreemanOrchestrator(client)

    package, _world_id, attempts, repair_history = orchestrator.compile_and_repair(
        "Repair an unstable package with escalating context.",
        max_retries=6,
        trial_steps=3,
    )

    assert package["schema"]["domain_id"] == malformed_schema["domain_id"]
    assert attempts >= 4
    repair_records = [record for record in repair_history if record.get("repair_stage")]
    assert repair_records[0]["repair_stage"] == "standard"
    assert repair_records[1]["repair_stage"] == "standard"
    assert repair_records[2]["repair_stage"] == "accumulated"
    assert client.feedback_log[0][0]["repair_stage"] == "standard"
    assert client.feedback_log[2][0]["repair_stage"] == "accumulated"
    assert client.feedback_log[2][0]["history_len"] >= 3


def test_package_normalization_moves_resource_relations_to_causal_dag() -> None:
    """Local-model normalization should repair the common actor/resource relation mixup."""

    malformed_package = {
        "schema": {
            "domain_id": "gas_demo",
            "actors": [
                {"id": "supplier", "name": "Supplier", "state": {"reliability": 0.8}},
                {"id": "consumer", "name": "Consumer", "state": {"stress": 0.2}},
            ],
            "resources": [
                {
                    "id": "gas_flow_bcm",
                    "name": "Gas Flow",
                    "value": 1.0,
                    "unit": "bcm_per_day",
                    "min_value": 0.0,
                    "max_value": 2.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.0, "b": 0.0, "c": 1.0, "coupling_weights": {}},
                },
                {
                    "id": "storage_fill_pct",
                    "name": "Storage Fill",
                    "value": 60.0,
                    "unit": "pct",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.0,
                        "b": 0.0,
                        "c": 60.0,
                        "coupling_weights": {"gas_flow_bcm": 0.1},
                    },
                },
            ],
            "relations": [
                {
                    "source_id": "gas_flow_bcm",
                    "target_id": "storage_fill_pct",
                    "relation_type": "drives",
                    "weights": {"storage_fill_pct": 0.1},
                }
            ],
            "outcomes": [
                {
                    "id": "storage_security",
                    "label": "Storage Security",
                    "scoring_weights": {"storage_fill_pct": 1.0},
                }
            ],
            "causal_dag": [],
        },
        "policies": [],
        "assumptions": [],
    }
    client = RepairingStubClient(malformed_package)
    orchestrator = FreemanOrchestrator(client, package_normalization="always")

    package = orchestrator._normalize_package(malformed_package)

    assert package["schema"]["relations"] == []
    assert package["schema"]["causal_dag"][0]["source"] == "gas_flow_bcm"
    assert package["schema"]["causal_dag"][0]["target"] == "storage_fill_pct"
    DomainCompiler().compile(package["schema"])


def test_package_normalization_can_be_disabled() -> None:
    """The sanitizer remains opt-in/auto and can be disabled for strict model evaluation."""

    malformed_package = {
        "schema": {
            "domain_id": "gas_demo",
            "actors": [{"id": "supplier", "name": "Supplier", "state": {}}],
            "resources": [
                {
                    "id": "gas_flow_bcm",
                    "name": "Gas Flow",
                    "value": 1.0,
                    "unit": "bcm_per_day",
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.0, "b": 0.0, "c": 1.0, "coupling_weights": {}},
                }
            ],
            "relations": [{"source_id": "supplier", "target_id": "gas_flow_bcm", "relation_type": "controls"}],
            "outcomes": [{"id": "flow", "label": "Flow", "scoring_weights": {"gas_flow_bcm": 1.0}}],
            "causal_dag": [],
        },
        "policies": [],
        "assumptions": [],
    }
    client = RepairingStubClient(malformed_package)
    orchestrator = FreemanOrchestrator(client, package_normalization="never")

    package = orchestrator._normalize_package(malformed_package)

    assert package["schema"]["relations"] == [
        {
            "source_id": "supplier",
            "target_id": "gas_flow_bcm",
            "relation_type": "controls",
            "weights": {},
        }
    ]


def test_normalize_resources_coerces_logistic_params_to_operator_contract() -> None:
    client = RepairingStubClient({"schema": {}, "policies": [], "assumptions": []})
    orchestrator = FreemanOrchestrator(client)

    resources = orchestrator._normalize_resources(
        [
            {
                "id": "warming_risk",
                "name": "Warming Risk",
                "value": 0.4,
                "max_value": 1.0,
                "evolution_type": "logistic",
                "evolution_params": {
                    "a": 0.8,
                    "b": 0.1,
                    "c": 0.05,
                    "coupling_weights": {"emissions": 0.2},
                },
            }
        ],
        actor_ids=set(),
    )

    params = resources[0]["evolution_params"]
    assert set(params) == {"r", "K", "external", "policy_scale", "coupling_weights"}
    assert "a" not in params
    assert params["K"] == 1.0
    assert params["coupling_weights"]["emissions"] == 0.2


def test_repair_stage_thresholds() -> None:
    orchestrator = FreemanOrchestrator(RepairingStubClient({"schema": {}, "policies": [], "assumptions": []}))

    assert orchestrator._repair_stage(1) == "standard"
    assert orchestrator._repair_stage(3) == "standard"
    assert orchestrator._repair_stage(4) == "accumulated"
    assert orchestrator._repair_stage(8) == "accumulated"
    assert orchestrator._repair_stage(9) == "schema_aware"


def test_normalize_package_canonicalizes_relation_aliases() -> None:
    orchestrator = FreemanOrchestrator(RepairingStubClient({"schema": {}, "policies": [], "assumptions": []}))

    package = orchestrator._normalize_package(
        {
            "schema": {
                "relations": [
                    {
                        "source": "a",
                        "target": "b",
                        "type": "trade",
                    }
                ]
            }
        }
    )

    assert package["schema"]["relations"] == [
        {
            "source_id": "a",
            "target_id": "b",
            "relation_type": "trade",
            "weights": {},
        }
    ]
