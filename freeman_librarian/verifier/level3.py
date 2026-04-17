"""Organization-specific Level 3 verifier invariants."""

from __future__ import annotations

from typing import Any

from freeman_librarian.core.types import Violation
from freeman_librarian.core.world import WorldState


def _org_metadata(world: WorldState) -> dict[str, Any]:
    metadata = dict(world.metadata.get("org", {}))
    return metadata if isinstance(metadata, dict) else {}


def level3_check(world: WorldState) -> list[Violation]:
    """Validate organization-specific structural invariants."""

    violations: list[Violation] = []
    org_metadata = _org_metadata(world)
    org_resources = dict(org_metadata.get("resources", {}))
    relation_payloads = list(org_metadata.get("relations", []))
    owner_count_by_process: dict[str, int] = {}
    for relation in relation_payloads:
        if relation.get("relation_type") != "owns":
            continue
        process_id = str(relation.get("target_id"))
        owner_count_by_process[process_id] = owner_count_by_process.get(process_id, 0) + 1

    process_resources = {
        resource_id: resource
        for resource_id, resource in world.resources.items()
        if str(org_resources.get(resource_id, {}).get("kind", "")) == "process"
    }
    for resource_id, resource in process_resources.items():
        owner_id = resource.owner_id
        if owner_id is None:
            violations.append(
                Violation(
                    level=3,
                    check_name="process_owner_missing",
                    description=f"Process {resource_id} has no owner.",
                    severity="hard",
                    details={"resource_id": resource_id},
                )
            )
        owner_count = int(owner_count_by_process.get(resource_id, 1))
        if owner_count != 1:
            violations.append(
                Violation(
                    level=3,
                    check_name="process_owner_cardinality",
                    description=f"Process {resource_id} must have exactly one owner, observed {owner_count}.",
                    severity="hard",
                    details={"resource_id": resource_id, "owner_count": owner_count},
                )
            )
        if owner_id is not None and owner_id not in world.actors:
            violations.append(
                Violation(
                    level=3,
                    check_name="process_owner_unknown_actor",
                    description=f"Process {resource_id} references unknown owner {owner_id}.",
                    severity="hard",
                    details={"resource_id": resource_id, "owner_id": owner_id},
                )
            )

    delegate_graph: dict[str, set[str]] = {}
    for relation in world.relations:
        if relation.relation_type != "delegates_to":
            continue
        delegate_graph.setdefault(relation.source_id, set()).add(relation.target_id)
    visited: set[str] = set()
    stack: set[str] = set()

    def _visit(node_id: str) -> bool:
        if node_id in stack:
            return True
        if node_id in visited:
            return False
        visited.add(node_id)
        stack.add(node_id)
        for neighbor in delegate_graph.get(node_id, set()):
            if _visit(neighbor):
                return True
        stack.remove(node_id)
        return False

    for actor_id in delegate_graph:
        if _visit(actor_id):
            violations.append(
                Violation(
                    level=3,
                    check_name="delegation_cycle",
                    description="Delegation graph contains a cycle.",
                    severity="hard",
                    details={"delegate_graph": {key: sorted(values) for key, values in delegate_graph.items()}},
                )
            )
            break

    referenced_actor_ids = set(org_metadata.get("referenced_actor_ids", []))
    missing_actor_ids = sorted(actor_id for actor_id in referenced_actor_ids if actor_id not in world.actors)
    if missing_actor_ids:
        violations.append(
            Violation(
                level=3,
                check_name="missing_referenced_roles",
                description="Some actor mentions from the corpus are absent from the actor graph.",
                severity="hard",
                details={"missing_actor_ids": missing_actor_ids},
            )
        )

    conflicts = list(org_metadata.get("conflicts", []))
    if conflicts:
        violations.append(
            Violation(
                level=3,
                check_name="document_conflicts_present",
                description="Conflicts between documents were detected and must be reconciled explicitly.",
                severity="soft",
                details={"conflicts": conflicts},
            )
        )

    return violations


__all__ = ["level3_check"]
