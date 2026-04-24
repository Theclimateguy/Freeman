"""LLM-driven domain synthesis and Freeman tool execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from freeman.api.tool_api import (
    freeman_compile_domain,
    freeman_get_world_state,
    freeman_run_simulation,
    freeman_verify_domain,
)
from freeman.core.transition import step_world
from freeman.core.types import Policy, Violation
from freeman.domain.compiler import DomainCompiler
from freeman.exceptions import HardStopException, SchemaRepairFailed
from freeman.game.runner import SimConfig
from freeman.utils import stable_json_dumps
from freeman.verifier.level1 import level1_check
from freeman.verifier.level2 import level2_check

SYNTHESIS_SYSTEM_PROMPT = """You are designing compact, valid Freeman simulation packages from natural-language domain briefs.

Return exactly one JSON object with these keys:
- schema: a valid Freeman domain schema
- policies: list of Freeman Policy snapshots
- assumptions: short list of modeling assumptions

Constraints:
- Keep the domain compact: 3-5 actors, 4-7 resources, 2-4 outcomes, 3-8 causal edges.
- Use only evolution types supported by Freeman: linear, stock_flow, logistic, threshold, coupled.
- Each relation object must use exactly these keys: source_id, target_id, relation_type, weights.
- scoring_weights may reference only resource ids or actor state keys already present in the schema.
- causal_dag signs must match the direction implied by the actual coupling weights.
- Set resource.conserved=true only for physically conserved stocks.
- Prefer explicit actor_update_rules over hidden metadata.
- Policies must use actor ids declared in the schema.
- Keep magnitudes moderate and numerically stable.
- Return JSON only.
"""

INTERPRETATION_SYSTEM_PROMPT = """You are analyzing a Freeman simulation result for an expert user.

Return exactly one JSON object with:
- dominant_outcome: string
- executive_summary: short paragraph
- key_dynamics: list of 3-5 concise points
- warnings: list of concise caveats
- suggested_next_policies: list of 2-4 policy ideas

Be concrete and tie every statement to the provided trajectory or verifier output.
Return JSON only.
"""


@dataclass
class LLMDrivenSimulationRun:
    """Serializable result of an LLM-synthesized Freeman simulation."""

    domain_description: str
    world_id: str
    synthesis: Dict[str, Any]
    simulation: Dict[str, Any]
    verification: Dict[str, Any]
    interpretation: Dict[str, Any]
    latest_world_state: Dict[str, Any]
    synthesis_attempts: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable view of the full run."""

        return {
            "domain_description": self.domain_description,
            "world_id": self.world_id,
            "synthesis": self.synthesis,
            "simulation": self.simulation,
            "verification": self.verification,
            "interpretation": self.interpretation,
            "latest_world_state": self.latest_world_state,
            "synthesis_attempts": self.synthesis_attempts,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize the run to deterministic JSON."""

        return stable_json_dumps(self.snapshot())


class FreemanOrchestrator:
    """Generate Freeman domain packages with an LLM client and execute them locally."""

    def __init__(self, client: Any, *, package_normalization: str | bool = "auto") -> None:
        self.client = client
        self.package_normalization = package_normalization
        self.last_bootstrap_attempts: List[Dict[str, Any]] = []

    def _synthesis_messages(self, domain_description: str) -> List[Dict[str, str]]:
        """Build the initial prompt sequence for Freeman package synthesis."""

        return [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Build a compact Freeman simulation package for this domain brief.\n\n"
                    f"{domain_description}\n\n"
                    "Use realistic assumptions, stable dynamics, and include a small baseline policy set "
                    "if interventions are meaningful. Return only the JSON package."
                ),
            },
        ]

    def _normalize_package(self, package: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure a package carries the standard top-level Freeman synthesis keys."""

        if not isinstance(package, dict):
            package = {}
        normalized = json.loads(json.dumps(package, ensure_ascii=False))
        if "schema" not in normalized and any(key in normalized for key in self._verifier_schema_spec()["required_top_level_keys"]):
            normalized = {
                "schema": normalized,
                "policies": [],
                "assumptions": [
                    "Input looked like a raw Freeman schema and was wrapped as a package.",
                ],
            }
        normalized.setdefault("schema", {})
        normalized.setdefault("policies", [])
        normalized.setdefault("assumptions", [])
        schema = normalized.get("schema", {})
        if self._package_normalization_enabled() and isinstance(schema, dict):
            normalized["schema"] = self._normalize_schema_payload(schema)
            return normalized
        relations = schema.get("relations", [])
        if isinstance(relations, list):
            normalized_relations: list[Dict[str, Any]] = []
            for relation in relations:
                if not isinstance(relation, dict):
                    normalized_relations.append(relation)
                    continue
                mapped = dict(relation)
                if "source_id" not in mapped and "source" in mapped:
                    mapped["source_id"] = mapped.pop("source")
                if "target_id" not in mapped and "target" in mapped:
                    mapped["target_id"] = mapped.pop("target")
                if "relation_type" not in mapped:
                    mapped["relation_type"] = mapped.pop("type", mapped.pop("label", "association"))
                mapped.setdefault("weights", {})
                normalized_relations.append(mapped)
            schema["relations"] = normalized_relations
        return normalized

    def _package_normalization_enabled(self) -> bool:
        """Return whether local LLM output should be normalized before verification."""

        mode = self.package_normalization
        if isinstance(mode, bool):
            return mode
        normalized = str(mode or "auto").strip().lower()
        if normalized in {"0", "false", "no", "off", "never", "none", "disabled"}:
            return False
        if normalized in {"1", "true", "yes", "on", "always", "force", "enabled"}:
            return True
        model = str(getattr(self.client, "model", "") or "").lower()
        if not model:
            return False
        if any(marker in model for marker in ("ollama", "qwen", "llama", "mistral", "coder")):
            return True
        match = re.search(r"(\d+(?:\.\d+)?)\s*b", model)
        return bool(match and float(match.group(1)) <= 14.0)

    def _normalize_schema_payload(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce common LLM schema mistakes into Freeman's compiler contract."""

        normalized = dict(schema)
        normalized["actors"] = self._normalize_actors(normalized.get("actors", []))
        actor_ids = {actor["id"] for actor in normalized["actors"]}

        normalized["resources"] = self._normalize_resources(normalized.get("resources", []), actor_ids=actor_ids)
        resource_ids = {resource["id"] for resource in normalized["resources"]}

        normalized["outcomes"] = self._normalize_outcomes(normalized.get("outcomes", []), resource_ids)
        actor_state_keys = self._actor_state_keys(normalized["actors"])
        valid_value_keys = set(resource_ids) | actor_state_keys

        relations, inferred_edges = self._normalize_relations(
            normalized.get("relations", []),
            actor_ids=actor_ids,
            valid_value_keys=valid_value_keys,
        )
        normalized["relations"] = relations
        normalized["causal_dag"] = self._normalize_causal_dag(
            [*self._as_list(normalized.get("causal_dag", [])), *inferred_edges],
            valid_value_keys=valid_value_keys,
        )
        normalized["actor_update_rules"] = self._normalize_actor_update_rules(
            normalized.get("actor_update_rules", {}),
            actor_ids=actor_ids,
            valid_value_keys=valid_value_keys,
        )
        return normalized

    def _normalize_actors(self, actors: Any) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        for index, actor in enumerate(self._as_list(actors), start=1):
            if isinstance(actor, str):
                actor = {"id": self._safe_id(actor, fallback=f"actor_{index}"), "name": actor, "state": {}}
            if not isinstance(actor, dict):
                continue
            actor_id = self._safe_id(actor.get("id") or actor.get("name"), fallback=f"actor_{index}")
            state = actor.get("state", {})
            if not isinstance(state, dict):
                state = {}
            values.append(
                {
                    "id": actor_id,
                    "name": str(actor.get("name") or actor_id.replace("_", " ").title()),
                    "state": {
                        self._safe_id(key, fallback=f"state_{idx}"): float(value)
                        for idx, (key, value) in enumerate(state.items(), start=1)
                        if self._is_number(value)
                    },
                    "metadata": actor.get("metadata", {}) if isinstance(actor.get("metadata", {}), dict) else {},
                }
            )
        return self._dedupe_by_id(values)

    def _normalize_resources(self, resources: Any, *, actor_ids: set[str]) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        for index, resource in enumerate(self._as_list(resources), start=1):
            if isinstance(resource, str):
                resource = {"id": self._safe_id(resource, fallback=f"resource_{index}"), "name": resource}
            if not isinstance(resource, dict):
                continue
            resource_id = self._safe_id(resource.get("id") or resource.get("name"), fallback=f"resource_{index}")
            params = resource.get("evolution_params", {})
            if not isinstance(params, dict):
                params = {}
            if not params:
                params = self._legacy_resource_params(resource)
            owner_id = resource.get("owner_id")
            evolution_type = str(resource.get("evolution_type") or "linear")
            if evolution_type not in {"linear", "stock_flow", "logistic", "threshold", "coupled"}:
                evolution_type = "linear"
            values.append(
                {
                    "id": resource_id,
                    "name": str(resource.get("name") or resource_id.replace("_", " ").title()),
                    "value": float(resource.get("value", resource.get("initial_value", 1.0)) or 0.0),
                    "unit": str(resource.get("unit") or resource.get("units") or "index"),
                    "owner_id": owner_id if owner_id in actor_ids else None,
                    "min_value": float(resource.get("min_value", 0.0) or 0.0),
                    "max_value": self._float_or_inf(resource.get("max_value", resource.get("capacity", float("inf")))),
                    "evolution_type": evolution_type,
                    "evolution_params": params,
                    "conserved": bool(resource.get("conserved", False)),
                }
            )
        return self._dedupe_by_id(values)

    def _normalize_outcomes(self, outcomes: Any, resource_ids: set[str]) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        for index, outcome in enumerate(self._as_list(outcomes), start=1):
            if isinstance(outcome, str):
                outcome = {"id": self._safe_id(outcome, fallback=f"outcome_{index}"), "label": outcome}
            if not isinstance(outcome, dict):
                continue
            outcome_id = self._safe_id(
                outcome.get("id") or outcome.get("label") or outcome.get("name"),
                fallback=f"outcome_{index}",
            )
            weights = outcome.get("scoring_weights", {})
            if not isinstance(weights, dict):
                weights = {}
            weights = {
                str(key): float(value)
                for key, value in weights.items()
                if key in resource_ids and self._is_number(value)
            }
            if not weights and resource_ids:
                weights = {sorted(resource_ids)[0]: 1.0}
            values.append(
                {
                    "id": outcome_id,
                    "label": str(
                        outcome.get("label")
                        or outcome.get("name")
                        or outcome_id.replace("_", " ").title()
                    ),
                    "scoring_weights": weights,
                    "description": str(outcome.get("description") or ""),
                    "regime_shifts": (
                        outcome.get("regime_shifts", [])
                        if isinstance(outcome.get("regime_shifts", []), list)
                        else []
                    ),
                }
            )
        return self._dedupe_by_id(values)

    def _normalize_relations(
        self,
        relations: Any,
        *,
        actor_ids: set[str],
        valid_value_keys: set[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        normalized_relations: List[Dict[str, Any]] = []
        inferred_edges: List[Dict[str, Any]] = []
        for relation in self._as_list(relations):
            if not isinstance(relation, dict):
                continue
            mapped = dict(relation)
            source = str(mapped.pop("source", mapped.get("source_id", "")))
            target = str(mapped.pop("target", mapped.get("target_id", "")))
            relation_type = str(mapped.pop("type", mapped.pop("label", mapped.get("relation_type", "association"))))
            weights = mapped.get("weights", {})
            weights = weights if isinstance(weights, dict) else {}
            if source in actor_ids and target in actor_ids:
                normalized_relations.append(
                    {
                        "source_id": source,
                        "target_id": target,
                        "relation_type": relation_type,
                        "weights": {str(key): float(value) for key, value in weights.items() if self._is_number(value)},
                    }
                )
                continue
            inferred_edges.extend(self._relations_to_causal_edges(source, target, weights, valid_value_keys))
        return normalized_relations, self._dedupe_causal_edges(inferred_edges)

    def _normalize_causal_dag(self, edges: Any, *, valid_value_keys: set[str]) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        for edge in self._as_list(edges):
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", edge.get("source_id", "")))
            target = str(edge.get("target", edge.get("target_id", "")))
            if source not in valid_value_keys or target not in valid_value_keys or source == target:
                continue
            expected_sign = str(edge.get("expected_sign") or edge.get("sign") or "+")
            expected_sign = "-" if expected_sign.strip().startswith("-") else "+"
            values.append(
                {
                    "source": source,
                    "target": target,
                    "expected_sign": expected_sign,
                    "strength": str(edge.get("strength") or "strong"),
                    "weight": None if edge.get("weight") is None else float(edge.get("weight")),
                    "weight_source": str(edge.get("weight_source") or "llm_normalized"),
                    "metadata": edge.get("metadata", {}) if isinstance(edge.get("metadata", {}), dict) else {},
                }
            )
        return self._dedupe_causal_edges(values)

    def _normalize_actor_update_rules(
        self,
        rules: Any,
        *,
        actor_ids: set[str],
        valid_value_keys: set[str],
    ) -> Dict[str, Any]:
        if not isinstance(rules, dict):
            return {}
        normalized: Dict[str, Any] = {}
        for actor_id, state_rules in rules.items():
            if actor_id not in actor_ids or not isinstance(state_rules, dict):
                continue
            cleaned_rules: Dict[str, Any] = {}
            for state_key, spec in state_rules.items():
                if not isinstance(spec, dict):
                    continue
                weights = spec.get("weights", {})
                if isinstance(weights, dict):
                    spec = dict(spec)
                    spec["weights"] = {
                        str(key): float(value)
                        for key, value in weights.items()
                        if key in valid_value_keys and self._is_number(value)
                    }
                cleaned_rules[str(state_key)] = spec
            if cleaned_rules:
                normalized[str(actor_id)] = cleaned_rules
        return normalized

    def _relations_to_causal_edges(
        self,
        source: str,
        target: str,
        weights: Dict[str, Any],
        valid_value_keys: set[str],
    ) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        if source in valid_value_keys and target in valid_value_keys and source != target:
            edges.append(self._causal_edge_from_weight(source, target, weights.get(target, 1.0)))
        if target in valid_value_keys:
            for key, value in weights.items():
                if key in valid_value_keys and key != target:
                    edges.append(self._causal_edge_from_weight(str(key), target, value))
        if source in valid_value_keys:
            for key, value in weights.items():
                if key in valid_value_keys and key != source:
                    edges.append(self._causal_edge_from_weight(source, str(key), value))
        return edges

    def _causal_edge_from_weight(self, source: str, target: str, weight: Any) -> Dict[str, Any]:
        value = float(weight) if self._is_number(weight) else 1.0
        return {
            "source": source,
            "target": target,
            "expected_sign": "-" if value < 0.0 else "+",
            "strength": "strong",
            "weight": value,
            "weight_source": "normalized_relation",
            "metadata": {"normalized_from": "relation"},
        }

    def _legacy_resource_params(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        if any(key in resource for key in ("inflow", "outflow", "growth_rate", "decay_rate")):
            growth = float(resource.get("growth_rate", 0.0) or 0.0)
            decay = float(resource.get("decay_rate", 0.0) or 0.0)
            inflow = float(resource.get("inflow", 0.0) or 0.0)
            outflow = float(resource.get("outflow", 0.0) or 0.0)
            return {
                "a": max(0.0, min(0.95, 1.0 + growth - decay)),
                "b": 0.0,
                "c": inflow - outflow,
                "coupling_weights": {},
            }
        return {"a": 0.8, "b": 0.0, "c": 0.0, "coupling_weights": {}}

    def _actor_state_keys(self, actors: List[Dict[str, Any]]) -> set[str]:
        keys: set[str] = set()
        for actor in actors:
            actor_id = actor["id"]
            for state_key in actor.get("state", {}):
                keys.add(str(state_key))
                keys.add(f"{actor_id}.{state_key}")
        return keys

    def _as_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _safe_id(self, value: Any, *, fallback: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
        return text or fallback

    def _dedupe_by_id(self, values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        result: List[Dict[str, Any]] = []
        for value in values:
            item_id = str(value.get("id", ""))
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            result.append(value)
        return result

    def _dedupe_causal_edges(self, values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        result: List[Dict[str, Any]] = []
        for value in values:
            key = (str(value.get("source", "")), str(value.get("target", "")))
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    def _is_number(self, value: Any) -> bool:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

    def _float_or_inf(self, value: Any) -> float:
        if value in {None, "inf", "infinity"}:
            return float("inf")
        return float(value)

    def _verifier_schema_spec(self) -> Dict[str, Any]:
        """Return the structural contract a package must satisfy to be verifier-clean."""

        return {
            "required_top_level_keys": ["domain_id", "actors", "resources", "relations", "outcomes", "causal_dag"],
            "supported_evolution_types": ["linear", "stock_flow", "logistic", "threshold", "coupled"],
            "compact_domain_bounds": {
                "actors": [3, 5],
                "resources": [4, 7],
                "outcomes": [2, 4],
                "causal_edges": [3, 8],
            },
            "repair_rules": [
                "actor ids and policy actor references must agree",
                "resource ids and scoring_weights references must agree",
                "relations must use source_id, target_id, relation_type, and weights",
                "causal_dag signs must match the effective coupling directions",
                "resource dynamics must remain numerically stable under level1/level2 verification",
                "return a full corrected package, not a patch",
            ],
        }

    def _coerce_policies(self, package: Dict[str, Any]) -> List[Policy]:
        """Convert synthesized policy snapshots into ``Policy`` objects."""

        return [policy if isinstance(policy, Policy) else Policy.from_snapshot(policy) for policy in package.get("policies", [])]

    def _compiler_feedback(self, exc: Exception) -> List[Dict[str, Any]]:
        """Convert a compile-time exception into structured repair feedback."""

        return [
            {
                "phase": "compile",
                "level": -1,
                "check_name": "compile_error",
                "description": str(exc),
                "severity": "hard",
                "details": {
                    "field": "schema",
                    "observed": str(exc),
                    "expected": "valid Freeman domain schema",
                    "error_type": exc.__class__.__name__,
                },
            }
        ]

    def _violation_feedback(self, phase: str, violations: List[Violation]) -> List[Dict[str, Any]]:
        """Serialize verifier violations into structured repair feedback."""

        return [{"phase": phase, **violation.snapshot()} for violation in violations]

    def _feedback_summary(self, feedback: List[Dict[str, Any]]) -> str:
        """Compress structured feedback into one deterministic summary line."""

        if not feedback:
            return "no verifier error"
        head = feedback[0]
        check_name = str(head.get("check_name", "unknown"))
        description = str(head.get("description", "")).strip()
        details = head.get("details", {}) or {}
        field_name = details.get("field") or details.get("repair_targets") or details.get("edge")
        if field_name:
            return f"{check_name}: {field_name} | {description}".strip()
        return f"{check_name}: {description}".strip()

    def _repair_stage(self, attempt: int) -> str:
        """Escalate repair context as attempts accumulate."""

        if attempt <= 3:
            return "standard"
        if attempt <= 8:
            return "accumulated"
        return "schema_aware"

    def _trial_level0_violations(
        self,
        package: Dict[str, Any],
        trial_steps: int,
        dt: float,
    ) -> List[Violation]:
        """Run a short rollout to surface level-0 violations quickly."""

        compiler = DomainCompiler()
        current = compiler.compile(package["schema"])
        policies = self._coerce_policies(package)
        violations: List[Violation] = []

        for _ in range(trial_steps):
            try:
                current, step_violations = step_world(current, policies, dt=dt)
            except HardStopException as exc:
                violations.extend(exc.violations)
                break
            violations.extend(step_violations)
        return violations

    def compile_and_repair(
        self,
        domain_description: str,
        *,
        max_retries: int = 5,
        trial_steps: int = 3,
        config: Optional[SimConfig] = None,
    ) -> tuple[Dict[str, Any], str, int, List[Dict[str, Any]]]:
        """Synthesize, verify, and iteratively repair a Freeman package until it compiles cleanly."""

        sim_config = config or SimConfig(convergence_check_steps=250, convergence_epsilon=3.0e-2)
        package = self._normalize_package(
            self.client.chat_json(self._synthesis_messages(domain_description), temperature=0.2, max_tokens=4000)
        )
        repair_history: List[Dict[str, Any]] = []
        self.last_bootstrap_attempts = []

        for attempt in range(1, max_retries + 1):
            try:
                world = DomainCompiler().compile(package["schema"])
            except Exception as exc:  # noqa: BLE001
                feedback = self._compiler_feedback(exc)
                attempt_record = {
                    "attempt": attempt,
                    "phase": "compile",
                    "verifier_error": self._feedback_summary(feedback),
                    "feedback": feedback,
                    "repair_stage": self._repair_stage(attempt + 1),
                }
                repair_history.append(attempt_record)
                self.last_bootstrap_attempts = json.loads(json.dumps(repair_history, ensure_ascii=False))
                package = self._normalize_package(
                    self.client.repair_schema(
                        package,
                        feedback,
                        domain_description=domain_description,
                        error_history=repair_history,
                        verifier_schema_spec=self._verifier_schema_spec(),
                        repair_stage=self._repair_stage(attempt + 1),
                    )
                )
                continue

            l1_violations = level1_check(world.clone(), sim_config)
            if l1_violations:
                feedback = self._violation_feedback("level1", l1_violations)
                attempt_record = {
                    "attempt": attempt,
                    "phase": "level1",
                    "verifier_error": self._feedback_summary(feedback),
                    "feedback": feedback,
                    "repair_stage": self._repair_stage(attempt + 1),
                }
                repair_history.append(attempt_record)
                self.last_bootstrap_attempts = json.loads(json.dumps(repair_history, ensure_ascii=False))
                package = self._normalize_package(
                    self.client.repair_schema(
                        package,
                        feedback,
                        domain_description=domain_description,
                        error_history=repair_history,
                        verifier_schema_spec=self._verifier_schema_spec(),
                        repair_stage=self._repair_stage(attempt + 1),
                    )
                )
                continue

            trial_violations = self._trial_level0_violations(package, trial_steps=trial_steps, dt=sim_config.dt)
            if trial_violations:
                feedback = self._violation_feedback("level0_trial", trial_violations)
                attempt_record = {
                    "attempt": attempt,
                    "phase": "level0_trial",
                    "verifier_error": self._feedback_summary(feedback),
                    "feedback": feedback,
                    "repair_stage": self._repair_stage(attempt + 1),
                }
                repair_history.append(attempt_record)
                self.last_bootstrap_attempts = json.loads(json.dumps(repair_history, ensure_ascii=False))
                package = self._normalize_package(
                    self.client.repair_schema(
                        package,
                        feedback,
                        domain_description=domain_description,
                        error_history=repair_history,
                        verifier_schema_spec=self._verifier_schema_spec(),
                        repair_stage=self._repair_stage(attempt + 1),
                    )
                )
                continue

            l2_violations = level2_check(
                world.clone(),
                world.causal_dag,
                base_delta=sim_config.level2_shock_delta,
                dt=sim_config.dt,
            )
            if l2_violations:
                feedback = self._violation_feedback("level2", l2_violations)
                attempt_record = {
                    "attempt": attempt,
                    "phase": "level2",
                    "verifier_error": self._feedback_summary(feedback),
                    "feedback": feedback,
                    "repair_stage": self._repair_stage(attempt + 1),
                }
                repair_history.append(attempt_record)
                self.last_bootstrap_attempts = json.loads(json.dumps(repair_history, ensure_ascii=False))
                package = self._normalize_package(
                    self.client.repair_schema(
                        package,
                        feedback,
                        domain_description=domain_description,
                        error_history=repair_history,
                        verifier_schema_spec=self._verifier_schema_spec(),
                        repair_stage=self._repair_stage(attempt + 1),
                    )
                )
                continue

            compile_result = freeman_compile_domain(package["schema"])
            self.last_bootstrap_attempts = json.loads(json.dumps(repair_history, ensure_ascii=False))
            return package, compile_result["world_id"], attempt, repair_history

        self.last_bootstrap_attempts = json.loads(json.dumps(repair_history, ensure_ascii=False))
        raise SchemaRepairFailed(f"LLM bootstrap did not produce a verifier-clean Freeman package after {max_retries} attempts.")

    def synthesize_package(self, domain_description: str, max_attempts: int = 3) -> tuple[Dict[str, Any], str, int]:
        """Use an LLM client to produce a verified Freeman schema and baseline policies."""

        package, world_id, attempts, _ = self.compile_and_repair(domain_description, max_retries=max_attempts)
        return package, world_id, attempts

    def interpret_run(
        self,
        domain_description: str,
        package: Dict[str, Any],
        simulation: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the configured LLM client to interpret the simulation outputs."""

        prompt_payload = {
            "domain_description": domain_description,
            "assumptions": package.get("assumptions", []),
            "final_outcome_probs": simulation.get("final_outcome_probs", {}),
            "steps_run": simulation.get("steps_run"),
            "confidence": simulation.get("confidence"),
            "violations": simulation.get("violations", []),
            "verification": verification,
            "trajectory_summary": self._summarize_trajectory(simulation),
        }
        messages = [
            {"role": "system", "content": INTERPRETATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ]
        return self.client.chat_json(messages, temperature=0.2, max_tokens=2500)

    def run(
        self,
        domain_description: str,
        *,
        max_steps: int = 20,
        seed: int = 42,
        verify_levels: Optional[List[int]] = None,
    ) -> LLMDrivenSimulationRun:
        """Synthesize a domain from text, run Freeman, and return the full result bundle."""

        package, world_id, attempts, repair_history = self.compile_and_repair(
            domain_description,
            max_retries=5,
            trial_steps=3,
            config=SimConfig(seed=seed, convergence_check_steps=250, convergence_epsilon=3.0e-2),
        )
        simulation = json.loads(
            freeman_run_simulation(
                world_id,
                package.get("policies", []),
                max_steps=max_steps,
                seed=seed,
            )
        )
        verification = freeman_verify_domain(world_id, levels=verify_levels or [1, 2])
        latest_world_state = freeman_get_world_state(world_id, t=-1)
        interpretation = self.interpret_run(domain_description, package, simulation, verification)
        return LLMDrivenSimulationRun(
            domain_description=domain_description,
            world_id=world_id,
            synthesis=package,
            simulation=simulation,
            verification=verification,
            interpretation=interpretation,
            latest_world_state=latest_world_state,
            synthesis_attempts=attempts,
            metadata={
                "model": self.client.model,
                "seed": seed,
                "max_steps": max_steps,
                "repair_iterations": max(0, attempts - 1),
                "repair_history": repair_history,
            },
        )
    def save_run(self, run: LLMDrivenSimulationRun, output_path: str | Path) -> Path:
        """Persist a run artifact to disk as JSON."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run.to_json(), encoding="utf-8")
        return path

    def _summarize_trajectory(self, simulation: Dict[str, Any]) -> Dict[str, Any]:
        """Compress the trajectory into start/end deltas for interpretation."""

        trajectory = simulation.get("trajectory", [])
        if len(trajectory) < 2:
            return {}
        first = trajectory[0]["resources"]
        last = trajectory[-1]["resources"]
        return {
            resource_id: {
                "start": first[resource_id]["value"],
                "end": last[resource_id]["value"],
                "delta": last[resource_id]["value"] - first[resource_id]["value"],
            }
            for resource_id in first
        }


DeepSeekFreemanOrchestrator = FreemanOrchestrator
