"""Multi-domain composition via shared resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from freeman_librarian.core.transition import step_world
from freeman_librarian.core.types import Policy, Resource, Violation
from freeman_librarian.core.world import WorldState
from freeman_librarian.utils import stable_json_dumps


@dataclass
class SharedResourceBus:
    """Mutable shared resource store used across multiple domains."""

    resources: Dict[str, Resource]

    def read(self, resource_id: str) -> float:
        """Read the current value of a shared resource."""

        return float(self.resources[resource_id].value)

    def write(self, resource_id: str, value: float) -> None:
        """Write a shared resource value."""

        self.resources[resource_id].value = value


@dataclass
class MultiDomainSimResult:
    """Serializable result of one multi-domain step."""

    domains: Dict[str, Dict]
    violations: List[Violation] = field(default_factory=list)
    shared_state: Dict[str, Dict] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the result to deterministic JSON."""

        return stable_json_dumps(
            {
                "domains": self.domains,
                "violations": [violation.snapshot() for violation in self.violations],
                "shared_state": self.shared_state,
            }
        )


class MultiDomainWorld:
    """Compose multiple worlds through a shared resource bus."""

    def __init__(self, domains: List[WorldState], shared_resource_ids: List[str]) -> None:
        self.domains = [domain.clone() for domain in domains]
        self.shared_resource_ids = list(shared_resource_ids)
        self.shared_bus = SharedResourceBus(
            resources={resource_id: self._find_resource(resource_id) for resource_id in self.shared_resource_ids}
        )

    def _find_resource(self, resource_id: str) -> Resource:
        """Find and clone a resource from the domain list."""

        for domain in self.domains:
            if resource_id in domain.resources:
                return Resource.from_snapshot(domain.resources[resource_id].snapshot())
        raise KeyError(f"Shared resource {resource_id} not found in any domain")

    def _sync_from_bus(self, domain: WorldState) -> None:
        """Copy shared resource values from the bus into a domain."""

        for resource_id in self.shared_resource_ids:
            if resource_id in domain.resources:
                domain.resources[resource_id].value = self.shared_bus.resources[resource_id].value

    def _sync_to_bus(self, domain: WorldState) -> None:
        """Copy shared resource values from a domain back to the bus."""

        for resource_id in self.shared_resource_ids:
            if resource_id in domain.resources:
                self.shared_bus.write(resource_id, domain.resources[resource_id].value)

    def step(self, policies: Dict[str, List[Policy]]) -> MultiDomainSimResult:
        """Advance each domain once while synchronizing shared resources."""

        all_violations: List[Violation] = []
        updated_domains: List[WorldState] = []

        for domain in self.domains:
            self._sync_from_bus(domain)
            next_domain, violations = step_world(domain, policies.get(domain.domain_id, []))
            self._sync_to_bus(next_domain)
            updated_domains.append(next_domain)
            all_violations.extend(violations)

        self.domains = updated_domains
        return MultiDomainSimResult(
            domains={domain.domain_id: domain.snapshot() for domain in self.domains},
            violations=all_violations,
            shared_state={resource_id: resource.snapshot() for resource_id, resource in self.shared_bus.resources.items()},
        )
