"""World state container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from freeman.core.types import Actor, CausalEdge, Outcome, Relation, Resource
from freeman.utils import deep_copy_jsonable, json_ready, normalize_numeric_tree


@dataclass
class WorldState:
    """Full simulator state at a discrete timestep."""

    domain_id: str
    t: int
    actors: Dict[str, Actor]
    resources: Dict[str, Resource]
    relations: List[Relation]
    outcomes: Dict[str, Outcome]
    causal_dag: List[CausalEdge]
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = int(self.t)
        self.seed = int(self.seed)
        self.metadata = normalize_numeric_tree(self.metadata)

    def snapshot(self) -> Dict[str, Any]:
        """Return a fully JSON-serializable snapshot of the world."""

        return {
            "domain_id": self.domain_id,
            "t": self.t,
            "actors": {actor_id: actor.snapshot() for actor_id, actor in self.actors.items()},
            "resources": {res_id: resource.snapshot() for res_id, resource in self.resources.items()},
            "relations": [relation.snapshot() for relation in self.relations],
            "outcomes": {outcome_id: outcome.snapshot() for outcome_id, outcome in self.outcomes.items()},
            "causal_dag": [edge.snapshot() for edge in self.causal_dag],
            "seed": self.seed,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "WorldState":
        """Recreate a world state from a snapshot."""

        return cls(
            domain_id=data["domain_id"],
            t=data["t"],
            actors={actor_id: Actor.from_snapshot(actor) for actor_id, actor in data["actors"].items()},
            resources={
                resource_id: Resource.from_snapshot(resource)
                for resource_id, resource in data["resources"].items()
            },
            relations=[Relation.from_snapshot(relation) for relation in data.get("relations", [])],
            outcomes={
                outcome_id: Outcome.from_snapshot(outcome)
                for outcome_id, outcome in data.get("outcomes", {}).items()
            },
            causal_dag=[CausalEdge.from_snapshot(edge) for edge in data.get("causal_dag", [])],
            seed=data.get("seed", 42),
            metadata=deep_copy_jsonable(data.get("metadata", {})),
        )

    def clone(self) -> "WorldState":
        """Return a deep copy of the world with no shared mutable state."""

        return WorldState.from_snapshot(self.snapshot())
