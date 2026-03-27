"""DeepSeek-driven domain synthesis and Freeman tool execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from freeman.api.tool_api import (
    freeman_compile_domain,
    freeman_get_world_state,
    freeman_run_simulation,
    freeman_verify_domain,
)
from freeman.llm.deepseek import DeepSeekChatClient
from freeman.utils import stable_json_dumps

SYNTHESIS_SYSTEM_PROMPT = """You are designing compact, valid Freeman simulation packages from natural-language domain briefs.

Return exactly one JSON object with these keys:
- schema: a valid Freeman domain schema
- policies: list of Freeman Policy snapshots
- assumptions: short list of modeling assumptions

Constraints:
- Keep the domain compact: 3-5 actors, 4-7 resources, 2-4 outcomes, 3-8 causal edges.
- Use only evolution types supported by Freeman: linear, stock_flow, logistic, threshold, coupled.
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


class DeepSeekFreemanOrchestrator:
    """Generate Freeman domain packages with DeepSeek and execute them locally."""

    def __init__(self, client: DeepSeekChatClient) -> None:
        self.client = client

    def synthesize_package(self, domain_description: str, max_attempts: int = 3) -> tuple[Dict[str, Any], str, int]:
        """Use DeepSeek to produce a valid Freeman schema and baseline policies."""

        messages: List[Dict[str, str]] = [
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

        last_error = None
        for attempt in range(1, max_attempts + 1):
            package = self.client.chat_json(messages, temperature=0.2, max_tokens=4000)
            try:
                compile_result = freeman_compile_domain(package["schema"])
                world_id = compile_result["world_id"]
                package.setdefault("policies", [])
                package.setdefault("assumptions", [])
                return package, world_id, attempt
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                messages.extend(
                    [
                        {"role": "assistant", "content": json.dumps(package, ensure_ascii=False)},
                        {
                            "role": "user",
                            "content": (
                                "The package was invalid for Freeman.\n"
                                f"Compiler/runtime error: {last_error}\n"
                                "Repair the full package and return only the corrected JSON object."
                            ),
                        },
                    ]
                )

        raise RuntimeError(f"DeepSeek did not produce a valid Freeman package after {max_attempts} attempts: {last_error}")

    def interpret_run(
        self,
        domain_description: str,
        package: Dict[str, Any],
        simulation: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask DeepSeek to interpret the simulation outputs."""

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

        package, world_id, attempts = self.synthesize_package(domain_description)
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
            metadata={"model": self.client.model, "seed": seed, "max_steps": max_steps},
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
