"""Minimal LLM-driven domain synthesis and one-step repair."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from freeman.core.transition import step_world
from freeman.core.types import Violation
from freeman.domain.compiler import DomainCompiler
from freeman.exceptions import HardStopException, SchemaRepairFailed
from freeman.game.runner import SimConfig
from freeman.verifier.verifier import Verifier

SYNTHESIS_SYSTEM_PROMPT = """You are designing compact, valid Freeman simulation packages from natural-language domain briefs.

Return exactly one JSON object with these keys:
- schema: a valid Freeman domain schema
- policies: list of Freeman Policy snapshots
- assumptions: short list of modeling assumptions

Constraints:
- Keep the domain compact and numerically stable.
- Use only evolution types supported by Freeman: linear, stock_flow, logistic, threshold, coupled.
- Each relation object must use exactly these keys: source_id, target_id, relation_type, weights.
- Return JSON only.
"""

REPAIR_SYSTEM_PROMPT = """You repair compact Freeman simulation packages from structured verifier feedback.

Return exactly one JSON object with:
- schema
- policies
- assumptions

Repair only what is needed to eliminate the reported violations. Return JSON only.
"""


@dataclass
class LLMDrivenSimulationRun:
    """Serializable synthesis payload kept for compatibility with older callers."""

    domain_description: str
    world_id: str
    synthesis: Dict[str, Any]
    verification: Dict[str, Any]
    synthesis_attempts: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "domain_description": self.domain_description,
            "world_id": self.world_id,
            "synthesis": self.synthesis,
            "verification": self.verification,
            "synthesis_attempts": self.synthesis_attempts,
            "metadata": self.metadata,
        }


class FreemanOrchestrator:
    """Generate a Freeman package and perform at most one repair pass."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.last_bootstrap_attempts: List[Dict[str, Any]] = []

    def _synthesis_messages(self, domain_description: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Build a compact Freeman simulation package for this domain brief.\n\n"
                    f"{domain_description}\n\n"
                    "Use realistic assumptions, stable dynamics, and return only the JSON package."
                ),
            },
        ]

    def _normalize_package(self, package: Dict[str, Any]) -> Dict[str, Any]:
        normalized = json.loads(json.dumps(package, ensure_ascii=False))
        normalized.setdefault("schema", {})
        normalized.setdefault("policies", [])
        normalized.setdefault("assumptions", [])
        relations = normalized.get("schema", {}).get("relations", [])
        normalized_relations: list[dict[str, Any]] = []
        for relation in relations if isinstance(relations, list) else []:
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
        normalized["schema"]["relations"] = normalized_relations
        return normalized

    def _compiler_feedback(self, exc: Exception) -> List[Dict[str, Any]]:
        return [
            {
                "phase": "compile",
                "level": -1,
                "check_name": "compile_error",
                "description": str(exc),
                "severity": "hard",
                "details": {"error_type": exc.__class__.__name__},
            }
        ]

    def _violation_feedback(self, phase: str, violations: List[Violation]) -> List[Dict[str, Any]]:
        return [{"phase": phase, **violation.snapshot()} for violation in violations]

    def _trial_level0_violations(
        self,
        package: Dict[str, Any],
        *,
        trial_steps: int,
        dt: float,
    ) -> List[Violation]:
        compiler = DomainCompiler()
        current = compiler.compile(package["schema"])
        violations: List[Violation] = []
        for _ in range(max(int(trial_steps), 0)):
            try:
                current, step_violations = step_world(current, [], dt=dt)
            except HardStopException as exc:
                violations.extend(exc.violations)
                break
            violations.extend(step_violations)
        return violations

    def _verification_feedback(
        self,
        package: Dict[str, Any],
        *,
        verify_level2: bool,
        trial_steps: int,
        config: SimConfig,
    ) -> tuple[str, List[Dict[str, Any]]]:
        try:
            world = DomainCompiler().compile(package["schema"])
        except Exception as exc:  # noqa: BLE001
            return "uncompiled", self._compiler_feedback(exc)

        trial_violations = self._trial_level0_violations(package, trial_steps=trial_steps, dt=config.dt)
        if trial_violations:
            return f"{world.domain_id}:{world.t}", self._violation_feedback("level0_trial", trial_violations)

        report = Verifier(config).run(world, levels=(1, 2) if verify_level2 else (1,))
        if report.violations:
            return f"{world.domain_id}:{world.t}", self._violation_feedback("verifier", list(report.violations))
        return f"{world.domain_id}:{world.t}", []

    def _repair_package(
        self,
        package: Dict[str, Any],
        violations: List[Dict[str, Any]],
        *,
        domain_description: str,
    ) -> Dict[str, Any]:
        if hasattr(self.client, "repair_schema"):
            repaired = self.client.repair_schema(
                package,
                violations,
                domain_description=domain_description,
                error_history=self.last_bootstrap_attempts,
            )
            return self._normalize_package(repaired)
        messages = [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "domain_description": domain_description,
                        "package": package,
                        "violations": violations,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self._normalize_package(self.client.chat_json(messages, temperature=0.1, max_tokens=4000))

    def compile_and_repair(
        self,
        domain_description: str,
        *,
        max_repairs: int = 1,
        trial_steps: int = 3,
        verify_level2: bool = False,
        config: SimConfig | None = None,
    ) -> tuple[Dict[str, Any], str, int, List[Dict[str, Any]]]:
        sim_config = config or SimConfig(convergence_check_steps=250, convergence_epsilon=3.0e-2)
        package = self._normalize_package(
            self.client.chat_json(self._synthesis_messages(domain_description), temperature=0.2, max_tokens=4000)
        )
        repair_history: List[Dict[str, Any]] = []
        self.last_bootstrap_attempts = []

        world_id, violations = self._verification_feedback(
            package,
            verify_level2=verify_level2,
            trial_steps=trial_steps,
            config=sim_config,
        )
        if not violations:
            return package, world_id, 1, repair_history

        if max_repairs <= 0:
            raise SchemaRepairFailed("LLM bootstrap did not produce a verifier-clean Freeman package.")

        repair_record = {"attempt": 2, "verifier_error": violations[0]["description"], "violations": violations}
        repair_history.append(repair_record)
        self.last_bootstrap_attempts = list(repair_history)
        repaired_package = self._repair_package(package, violations, domain_description=domain_description)
        repaired_world_id, repaired_violations = self._verification_feedback(
            repaired_package,
            verify_level2=verify_level2,
            trial_steps=trial_steps,
            config=sim_config,
        )
        if repaired_violations:
            raise SchemaRepairFailed(
                "LLM bootstrap did not produce a verifier-clean Freeman package after one repair pass."
            )
        return repaired_package, repaired_world_id, 2, repair_history

    def synthesize_package(self, domain_description: str, max_attempts: int = 2) -> tuple[Dict[str, Any], str, int]:
        package, world_id, attempts, _ = self.compile_and_repair(
            domain_description,
            max_repairs=max(max_attempts - 1, 0),
        )
        return package, world_id, attempts


DeepSeekFreemanOrchestrator = FreemanOrchestrator

__all__ = ["DeepSeekFreemanOrchestrator", "FreemanOrchestrator", "LLMDrivenSimulationRun"]
