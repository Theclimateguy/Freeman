"""World graph container and outcome registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

import numpy as np

from freeman.core.types import Actor, CausalEdge, Outcome, ParameterVector, Relation, Resource
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
    parameter_vector: ParameterVector = field(default_factory=ParameterVector)
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
        if isinstance(self.parameter_vector, dict):
            self.parameter_vector = ParameterVector.from_snapshot(self.parameter_vector)
        elif not isinstance(self.parameter_vector, ParameterVector):
            self.parameter_vector = ParameterVector()

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

    def edges(self, *, as_objects: bool = False) -> List[Any]:
        """Return causal edges as tuples by default or as ``CausalEdge`` objects."""

        if as_objects:
            return list(self.causal_dag)
        return [(edge.source, edge.target) for edge in self.causal_dag]

    def update_edge_weights(
        self,
        weights: Mapping[tuple[str, str], float] | Any,
        source: str = "causal_estimate",
        *,
        confidence_intervals: Optional[Mapping[tuple[str, str], tuple[float, float]]] = None,
        metadata: Optional[Mapping[tuple[str, str], Dict[str, Any]]] = None,
    ) -> Dict[tuple[str, str], float]:
        """Apply numeric causal-edge weights and annotate provenance.

        The update writes both to the declarative ``causal_dag`` edge metadata and,
        when possible, to the target resource's ``coupling_weights`` so the simulator
        uses the new magnitude in subsequent transitions.
        """

        resolved_weights = getattr(weights, "weights", weights)
        if not isinstance(resolved_weights, Mapping):
            raise TypeError("weights must be a mapping or an EstimationResult-like object with `.weights`")

        ci_map = dict(confidence_intervals or getattr(weights, "confidence_intervals", {}) or {})
        metadata_map = dict(metadata or getattr(weights, "edge_metadata", {}) or {})
        updates: Dict[tuple[str, str], float] = {}

        for edge in self.causal_dag:
            edge_key = (edge.source, edge.target)
            if edge_key not in resolved_weights:
                continue

            estimate = float(np.float64(resolved_weights[edge_key]))
            edge.weight = np.float64(estimate)
            edge.weight_source = str(source)
            edge.weight_confidence_interval = ci_map.get(edge_key)
            edge.metadata = {
                **dict(edge.metadata),
                **dict(metadata_map.get(edge_key, {})),
                "weight_source": str(source),
            }
            self._apply_resource_coupling_weight(edge.target, edge.source, estimate)
            updates[edge_key] = estimate

        if updates:
            update_log = self.metadata.setdefault("_causal_edge_updates", {})
            for (edge_source, edge_target), estimate in updates.items():
                payload: Dict[str, Any] = {"weight": estimate, "source": str(source)}
                if (edge_source, edge_target) in ci_map:
                    low, high = ci_map[(edge_source, edge_target)]
                    payload["confidence_interval"] = [float(low), float(high)]
                if (edge_source, edge_target) in metadata_map:
                    payload["metadata"] = deep_copy_jsonable(metadata_map[(edge_source, edge_target)])
                update_log[f"{edge_source}->{edge_target}"] = payload

        return updates

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
            "parameter_vector": self.parameter_vector.snapshot(),
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
            parameter_vector=ParameterVector.from_snapshot(data.get("parameter_vector", {})),
        )

    def clone(self) -> "WorldGraph":
        """Return a deep copy of the world with no shared mutable state."""

        return type(self).from_snapshot(self.snapshot())

    def apply_shocks(
        self,
        resource_shocks: Dict[str, float] | None = None,
        *,
        actor_state_shocks: Dict[str, float] | None = None,
        metadata_shocks: Dict[str, float] | None = None,
        time_decay: float = 1.0,
    ) -> "WorldGraph":
        """Apply stateful shocks on top of a decayed prior state.

        The method preserves a baseline snapshot inside ``metadata["_baseline_state"]``.
        Before applying new shocks, current deviations from that baseline are multiplied
        by ``time_decay``. New shocks are then added at full strength.
        """

        updated = self.clone()
        baseline = updated._ensure_baseline_state()
        shock_state = updated._ensure_shock_state(baseline)
        decay = np.float64(time_decay) * np.float64(self.parameter_vector.shock_decay)
        updated._decay_toward_baseline(baseline, decay)
        updated._decay_shock_state(shock_state, decay)
        updated._apply_resource_shocks(resource_shocks or {})
        updated._apply_actor_state_shocks(actor_state_shocks or {})
        updated._apply_metadata_shocks(metadata_shocks or {})
        return updated

    def _ensure_baseline_state(self) -> Dict[str, Any]:
        """Persist and return the immutable baseline state for decay calculations."""

        baseline = self.metadata.get("_baseline_state")
        if isinstance(baseline, dict):
            return deep_copy_jsonable(baseline)

        baseline = {
            "resources": {resource_id: float(resource.value) for resource_id, resource in self.resources.items()},
            "actors": {
                actor_id: {key: float(value) for key, value in actor.state.items()}
                for actor_id, actor in self.actors.items()
            },
            "metadata": {
                key: float(value)
                for key, value in self.metadata.items()
                if not str(key).startswith("_") and isinstance(value, (int, float, np.generic))
            },
        }
        self.metadata["_baseline_state"] = deep_copy_jsonable(baseline)
        return deep_copy_jsonable(baseline)

    def _decay_toward_baseline(self, baseline: Dict[str, Any], time_decay: np.float64) -> None:
        """Shrink all stored deviations toward the baseline state."""

        resource_baseline = baseline.get("resources", {})
        for resource_id, resource in self.resources.items():
            base_value = np.float64(resource_baseline.get(resource_id, float(resource.value)))
            decayed = base_value + (np.float64(resource.value) - base_value) * time_decay
            resource.value = np.float64(min(max(decayed, resource.min_value), resource.max_value))

        actor_baseline = baseline.get("actors", {})
        for actor_id, actor in self.actors.items():
            base_state = actor_baseline.get(actor_id, {})
            for key, value in actor.state.items():
                base_value = np.float64(base_state.get(key, float(value)))
                actor.state[key] = np.float64(base_value + (np.float64(value) - base_value) * time_decay)

        metadata_baseline = baseline.get("metadata", {})
        for key, value in list(self.metadata.items()):
            if str(key).startswith("_") or not isinstance(value, (int, float, np.generic)):
                continue
            base_value = np.float64(metadata_baseline.get(key, float(value)))
            self.metadata[key] = np.float64(base_value + (np.float64(value) - base_value) * time_decay)

    def _ensure_shock_state(self, baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Persist and return the accumulated shock/deviation state."""

        shock_state = self.metadata.get("_shock_state")
        if isinstance(shock_state, dict):
            return deep_copy_jsonable(shock_state)

        resource_baseline = baseline.get("resources", {})
        actor_baseline = baseline.get("actors", {})
        metadata_baseline = baseline.get("metadata", {})
        shock_state = {
            "resources": {
                resource_id: float(np.float64(resource.value) - np.float64(resource_baseline.get(resource_id, float(resource.value))))
                for resource_id, resource in self.resources.items()
            },
            "actors": {
                actor_id: {
                    key: float(np.float64(value) - np.float64(actor_baseline.get(actor_id, {}).get(key, float(value))))
                    for key, value in actor.state.items()
                }
                for actor_id, actor in self.actors.items()
            },
            "metadata": {
                key: float(np.float64(value) - np.float64(metadata_baseline.get(key, float(value))))
                for key, value in self.metadata.items()
                if not str(key).startswith("_") and isinstance(value, (int, float, np.generic))
            },
        }
        self.metadata["_shock_state"] = deep_copy_jsonable(shock_state)
        return deep_copy_jsonable(shock_state)

    def _decay_shock_state(self, shock_state: Dict[str, Any], time_decay: np.float64) -> None:
        """Decay the stored shock/deviation state in-place."""

        decayed_resources = {
            resource_id: float(np.float64(delta) * time_decay)
            for resource_id, delta in shock_state.get("resources", {}).items()
        }
        decayed_actors = {
            actor_id: {
                key: float(np.float64(delta) * time_decay)
                for key, delta in actor_state.items()
            }
            for actor_id, actor_state in shock_state.get("actors", {}).items()
        }
        decayed_metadata = {
            key: float(np.float64(delta) * time_decay)
            for key, delta in shock_state.get("metadata", {}).items()
        }
        self.metadata["_shock_state"] = {
            "resources": decayed_resources,
            "actors": decayed_actors,
            "metadata": decayed_metadata,
        }

    def _apply_resource_shocks(self, resource_shocks: Dict[str, float]) -> None:
        """Apply additive resource shocks with resource bounds."""

        shock_state = self.metadata.setdefault("_shock_state", {"resources": {}, "actors": {}, "metadata": {}})
        resource_state = shock_state.setdefault("resources", {})
        for resource_id, delta in resource_shocks.items():
            resource = self.resources.get(resource_id)
            if resource is None:
                raise KeyError(f"Unknown resource shock key: {resource_id}")
            new_value = np.float64(resource.value) + np.float64(delta)
            resource.value = np.float64(min(max(new_value, resource.min_value), resource.max_value))
            resource_state[resource_id] = float(np.float64(resource_state.get(resource_id, 0.0)) + np.float64(delta))

    def _apply_actor_state_shocks(self, actor_state_shocks: Dict[str, float]) -> None:
        """Apply additive shocks to actor state fields.

        Keys must be of the form ``actor_id.state_key``.
        """

        shock_state = self.metadata.setdefault("_shock_state", {"resources": {}, "actors": {}, "metadata": {}})
        actor_state = shock_state.setdefault("actors", {})
        for key, delta in actor_state_shocks.items():
            if "." not in key:
                raise KeyError(f"Actor-state shock keys must use actor_id.state_key format: {key}")
            actor_id, state_key = key.split(".", 1)
            actor = self.actors.get(actor_id)
            if actor is None:
                raise KeyError(f"Unknown actor in shock key: {key}")
            current = np.float64(actor.state.get(state_key, 0.0))
            actor.state[state_key] = np.float64(current + np.float64(delta))
            actor_state.setdefault(actor_id, {})
            actor_state[actor_id][state_key] = float(
                np.float64(actor_state[actor_id].get(state_key, 0.0)) + np.float64(delta)
            )

    def _apply_metadata_shocks(self, metadata_shocks: Dict[str, float]) -> None:
        """Apply additive shocks to numeric world metadata fields."""

        shock_state = self.metadata.setdefault("_shock_state", {"resources": {}, "actors": {}, "metadata": {}})
        metadata_state = shock_state.setdefault("metadata", {})
        for key, delta in metadata_shocks.items():
            if str(key).startswith("_"):
                raise KeyError(f"Reserved metadata key cannot be shocked: {key}")
            current = self.metadata.get(key, 0.0)
            if not isinstance(current, (int, float, np.generic)):
                current = 0.0
            self.metadata[key] = np.float64(current) + np.float64(delta)
            metadata_state[key] = float(np.float64(metadata_state.get(key, 0.0)) + np.float64(delta))

    def _apply_resource_coupling_weight(self, target_key: str, source_key: str, weight: float) -> None:
        """Write a numeric edge weight into the target resource evolution parameters."""

        resource = self.resources.get(target_key)
        if resource is None:
            return

        params = resource.evolution_params
        matched_paths = self._set_coupling_weight_recursive(params, source_key, np.float64(weight))
        if matched_paths:
            return

        coupling_weights = params.setdefault("coupling_weights", {})
        coupling_weights[source_key] = np.float64(weight)

    def _set_coupling_weight_recursive(self, node: Any, source_key: str, weight: np.float64) -> int:
        """Recursively update matching ``coupling_weights`` entries in nested params."""

        matches = 0
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "coupling_weights" and isinstance(value, dict):
                    if source_key in value:
                        value[source_key] = np.float64(weight)
                        matches += 1
                    continue
                matches += self._set_coupling_weight_recursive(value, source_key, weight)
        elif isinstance(node, list):
            for item in node:
                matches += self._set_coupling_weight_recursive(item, source_key, weight)
        return matches


WorldState = WorldGraph

__all__ = ["OutcomeRegistry", "WorldGraph", "WorldState"]
