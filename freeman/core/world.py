"""World graph container and outcome registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

from freeman.core.types import Actor, CausalEdge, Outcome, Relation, Resource
from freeman.utils import deep_copy_jsonable, json_ready, normalize_numeric_tree


@dataclass
class OutcomeRegistry:
    """Mutable registry of named outcomes used by the scorer."""

    outcomes: Dict[str, Outcome] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.outcomes = dict(self.outcomes)

    def add(self, outcome: Outcome) -> None:
        """Register or replace an outcome by id."""

        self.outcomes[outcome.id] = outcome

    def extend(self, outcomes: Iterable[Outcome]) -> None:
        """Register a collection of outcomes."""

        for outcome in outcomes:
            self.add(outcome)

    def get(self, outcome_id: str) -> Optional[Outcome]:
        """Return an outcome by id when present."""

        return self.outcomes.get(outcome_id)

    def remove(self, outcome_id: str) -> Outcome:
        """Remove an outcome and return it."""

        return self.outcomes.pop(outcome_id)

    def items(self) -> Iterator[tuple[str, Outcome]]:
        """Iterate over registered outcomes."""

        return iter(self.outcomes.items())

    def values(self) -> Iterator[Outcome]:
        """Iterate over registered outcome values."""

        return iter(self.outcomes.values())

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable registry snapshot."""

        return {outcome_id: outcome.snapshot() for outcome_id, outcome in self.outcomes.items()}

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "OutcomeRegistry":
        """Recreate a registry from a snapshot payload."""

        return cls(
            outcomes={
                outcome_id: Outcome.from_snapshot(outcome)
                for outcome_id, outcome in data.items()
            }
        )

    def clone(self) -> "OutcomeRegistry":
        """Return a deep copy of the registry."""

        return OutcomeRegistry.from_snapshot(self.snapshot())


@dataclass
class WorldGraph:
    """Full simulator state at a discrete timestep."""

    domain_id: str
    t: int
    actors: Dict[str, Actor]
    resources: Dict[str, Resource]
    relations: List[Relation]
    outcomes: Dict[str, Outcome] | OutcomeRegistry
    causal_dag: List[CausalEdge]
    actor_update_rules: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)
    _outcome_registry: OutcomeRegistry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.t = int(self.t)
        self.seed = int(self.seed)
        if isinstance(self.outcomes, OutcomeRegistry):
            self._outcome_registry = self.outcomes.clone()
        else:
            self._outcome_registry = OutcomeRegistry(self.outcomes)
        self.outcomes = self._outcome_registry.outcomes
        self.actor_update_rules = normalize_numeric_tree(self.actor_update_rules)
        self.metadata = normalize_numeric_tree(self.metadata)

    @property
    def outcome_registry(self) -> OutcomeRegistry:
        """Return the canonical outcome registry."""

        return self._outcome_registry

    def add_actor(self, actor: Actor) -> None:
        """Register or replace an actor by id."""

        self.actors[actor.id] = actor

    def add_resource(self, resource: Resource) -> None:
        """Register or replace a resource by id."""

        self.resources[resource.id] = resource

    def add_relation(self, relation: Relation) -> None:
        """Append a relation to the graph."""

        self.relations.append(relation)

    def add_outcome(self, outcome: Outcome) -> None:
        """Register or replace an outcome in the registry."""

        self._outcome_registry.add(outcome)

    def snapshot(self) -> Dict[str, Any]:
        """Return a fully JSON-serializable snapshot of the world."""

        return {
            "domain_id": self.domain_id,
            "t": self.t,
            "actors": {actor_id: actor.snapshot() for actor_id, actor in self.actors.items()},
            "resources": {res_id: resource.snapshot() for res_id, resource in self.resources.items()},
            "relations": [relation.snapshot() for relation in self.relations],
            "outcomes": self._outcome_registry.snapshot(),
            "causal_dag": [edge.snapshot() for edge in self.causal_dag],
            "actor_update_rules": json_ready(self.actor_update_rules),
            "seed": self.seed,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "WorldGraph":
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
            actor_update_rules=deep_copy_jsonable(data.get("actor_update_rules", {})),
            seed=data.get("seed", 42),
            metadata=deep_copy_jsonable(data.get("metadata", {})),
        )

    def clone(self) -> "WorldGraph":
        """Return a deep copy of the world with no shared mutable state."""

        return type(self).from_snapshot(self.snapshot())


WorldState = WorldGraph

__all__ = ["OutcomeRegistry", "WorldGraph", "WorldState"]
