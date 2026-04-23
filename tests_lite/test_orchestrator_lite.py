from __future__ import annotations

import copy
from typing import Any, Dict, List

from freeman.llm.orchestrator import FreemanOrchestrator


class RepairingStubClient:
    def __init__(self, initial_package: Dict[str, Any], repaired_package: Dict[str, Any]) -> None:
        self.initial_package = copy.deepcopy(initial_package)
        self.repaired_package = copy.deepcopy(repaired_package)
        self.feedback_log: List[List[Dict[str, Any]]] = []

    def chat_json(self, messages, *, temperature: float = 0.2, max_tokens: int | None = None) -> Dict[str, Any]:
        del messages, temperature, max_tokens
        return copy.deepcopy(self.initial_package)

    def repair_schema(
        self,
        package: Dict[str, Any],
        violations: List[Dict[str, Any]],
        *,
        domain_description: str = "",
        error_history: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        del package, domain_description, error_history
        self.feedback_log.append(copy.deepcopy(violations))
        return copy.deepcopy(self.repaired_package)


def test_orchestrator_repairs_once(water_market_schema: Dict[str, Any]) -> None:
    broken_schema = copy.deepcopy(water_market_schema)
    for resource in broken_schema["resources"]:
        if resource["id"] == "agriculture_output":
            resource["evolution_params"]["a"] = 1.25
            resource["evolution_params"]["coupling_weights"]["water_stock"] = -0.2

    client = RepairingStubClient(
        {"schema": broken_schema, "policies": [], "assumptions": ["Broken on purpose."]},
        {"schema": water_market_schema, "policies": [], "assumptions": ["Repaired."]},
    )
    orchestrator = FreemanOrchestrator(client)

    package, world_id, attempts, repair_history = orchestrator.compile_and_repair(
        "Repair a malformed water-market package.",
        max_repairs=1,
    )

    assert package["schema"]["domain_id"] == "water_market"
    assert world_id.startswith("water_market:")
    assert attempts == 2
    assert len(repair_history) == 1
    assert client.feedback_log
