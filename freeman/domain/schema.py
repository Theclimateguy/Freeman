"""Schema helpers for Freeman domains."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Set

from freeman.exceptions import ValidationError

REQUIRED_SCHEMA_KEYS = {
    "domain_id",
    "actors",
    "resources",
    "relations",
    "outcomes",
    "causal_dag",
}

SPATIAL_REGION_NODE_TYPE = "SpatialRegion"
GEO_HAS_GEOMETRY = "geo:hasGeometry"
GEO_BORDERED_BY = "geo:borderedBy"
GEO_WITHIN_REGION = "geo:withinRegion"
GEO_INTERSECTS = "geo:intersects"
GEO_HAS_CLIMATE_ZONE = "geo:hasClimateZone"
GEO_PREDICATES = {
    GEO_HAS_GEOMETRY,
    GEO_BORDERED_BY,
    GEO_WITHIN_REGION,
    GEO_INTERSECTS,
    GEO_HAS_CLIMATE_ZONE,
}


def validate_required_keys(schema: Dict[str, Any]) -> None:
    """Validate the presence of top-level required schema keys."""

    missing = sorted(REQUIRED_SCHEMA_KEYS - set(schema))
    if missing:
        raise ValidationError(f"Missing required schema keys: {', '.join(missing)}")


def ensure_unique_ids(items: Iterable[Dict[str, Any]], item_name: str) -> None:
    """Ensure that each item in a schema collection has a unique ``id`` field."""

    seen: Set[str] = set()
    duplicates: Set[str] = set()
    for item in items:
        item_id = item.get("id")
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    if duplicates:
        duplicate_str = ", ".join(sorted(str(item_id) for item_id in duplicates))
        raise ValidationError(f"Duplicate {item_name} ids: {duplicate_str}")


def collect_actor_state_keys(schema: Dict[str, Any]) -> Set[str]:
    """Return all actor-state keys accepted by outcome and causal references."""

    keys: Set[str] = set()
    for actor in schema.get("actors", []):
        actor_id = actor["id"]
        for state_key in actor.get("state", {}):
            keys.add(state_key)
            keys.add(f"{actor_id}.{state_key}")
    return keys
