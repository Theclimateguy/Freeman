"""Domain compiler from JSON schema to ``WorldState``."""

from __future__ import annotations

from typing import Any, Dict, Set

from freeman.core.evolution import EVOLUTION_REGISTRY
from freeman.core.types import Actor, CausalEdge, Outcome, Relation, Resource
from freeman.core.world import WorldState
from freeman.domain.schema import collect_actor_state_keys, ensure_unique_ids, validate_required_keys
from freeman.exceptions import ValidationError
from freeman.utils import deep_copy_jsonable


class DomainCompiler:
    """Compile and validate Freeman domain schemas."""

    def compile(self, schema: Dict[str, Any]) -> WorldState:
        """Validate ``schema`` and return an initialized ``WorldState``."""

        self._validate_schema(schema)
        metadata = deep_copy_jsonable(schema.get("metadata", {}))
        if "name" in schema:
            metadata["name"] = schema["name"]
        if "description" in schema:
            metadata["description"] = schema["description"]
        if "exogenous_inflows" in schema:
            metadata["exogenous_inflows"] = deep_copy_jsonable(schema["exogenous_inflows"])

        return WorldState(
            domain_id=schema["domain_id"],
            t=0,
            actors={actor["id"]: Actor(**actor) for actor in schema.get("actors", [])},
            resources={resource["id"]: Resource(**resource) for resource in schema.get("resources", [])},
            relations=[Relation(**relation) for relation in schema.get("relations", [])],
            outcomes={outcome["id"]: Outcome(**outcome) for outcome in schema.get("outcomes", [])},
            causal_dag=[CausalEdge(**edge) for edge in schema.get("causal_dag", [])],
            seed=schema.get("seed", 42),
            metadata=metadata,
        )

    def _validate_schema(self, schema: Dict[str, Any]) -> None:
        """Validate references, operator types, and score keys inside a domain schema."""

        validate_required_keys(schema)
        ensure_unique_ids(schema.get("actors", []), "actor")
        ensure_unique_ids(schema.get("resources", []), "resource")
        ensure_unique_ids(schema.get("outcomes", []), "outcome")

        actor_ids = {actor["id"] for actor in schema.get("actors", [])}
        resource_ids = {resource["id"] for resource in schema.get("resources", [])}
        actor_state_keys = collect_actor_state_keys(schema)

        for resource in schema.get("resources", []):
            owner_id = resource.get("owner_id")
            if owner_id is not None and owner_id not in actor_ids:
                raise ValidationError(f"Resource {resource['id']} references unknown owner_id {owner_id}")
            evolution_type = resource.get("evolution_type", "stock_flow")
            if evolution_type not in EVOLUTION_REGISTRY:
                raise ValidationError(f"Unknown evolution_type {evolution_type} for resource {resource['id']}")

        for relation in schema.get("relations", []):
            if relation["source_id"] not in actor_ids:
                raise ValidationError(f"Relation source {relation['source_id']} does not exist")
            if relation["target_id"] not in actor_ids:
                raise ValidationError(f"Relation target {relation['target_id']} does not exist")

        valid_value_keys = self._collect_valid_value_keys(resource_ids, actor_state_keys)
        for outcome in schema.get("outcomes", []):
            for key in outcome.get("scoring_weights", {}):
                if key not in valid_value_keys:
                    raise ValidationError(f"Outcome {outcome['id']} references unknown scoring key {key}")

        for edge in schema.get("causal_dag", []):
            if edge["source"] not in valid_value_keys:
                raise ValidationError(f"Causal edge source {edge['source']} does not exist")
            if edge["target"] not in valid_value_keys:
                raise ValidationError(f"Causal edge target {edge['target']} does not exist")

        inflows = schema.get("exogenous_inflows", {})
        for resource_id in inflows:
            if resource_id not in resource_ids:
                raise ValidationError(f"Exogenous inflow references unknown resource {resource_id}")

    def _collect_valid_value_keys(self, resource_ids: Set[str], actor_state_keys: Set[str]) -> Set[str]:
        """Return all keys that may legally reference world values."""

        return set(resource_ids) | set(actor_state_keys)
