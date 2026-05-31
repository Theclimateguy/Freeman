"""Spatial metadata materialization for compiled Freeman worlds."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

from freeman.core.types import Relation
from freeman.core.world import WorldState

LOGGER = logging.getLogger(__name__)
SPATIAL_NEIGHBOR_RELATION = "spatial_neighbor"
SPATIAL_REGION_WARNING_THRESHOLD = 100
SPATIAL_RELATION_WARNING_THRESHOLD = 10_000
_SPATIAL_ACTOR_REGION_KEYS = (
    "geo_id",
    "region_id",
    "spatial_region",
    "spatial_region_id",
    "iso_a3",
    "ISO_A3",
)
_SPATIAL_ACTOR_MAP_KEYS = (
    "actor_regions",
    "actor_region_map",
    "actor_to_region",
    "region_by_actor",
)


def _spatial_metadata(world: WorldState) -> dict[str, Any]:
    spatial = world.metadata.get("spatial", {})
    return spatial if isinstance(spatial, dict) else {}


def _coerce_region_id(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("geo_id", "region_id", "id"):
            if key in value:
                return _coerce_region_id(value[key])
        return None
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _metadata_region_id(metadata: dict[str, Any]) -> str | None:
    for key in _SPATIAL_ACTOR_REGION_KEYS:
        if key in metadata:
            region_id = _coerce_region_id(metadata[key])
            if region_id:
                return region_id
    geo_payload = metadata.get("geo")
    if isinstance(geo_payload, dict):
        return _metadata_region_id(geo_payload)
    spatial_payload = metadata.get("spatial")
    if isinstance(spatial_payload, dict):
        return _metadata_region_id(spatial_payload)
    return None


def _actor_region_map(world: WorldState, spatial: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key in _SPATIAL_ACTOR_MAP_KEYS:
        payload = spatial.get(key)
        if not isinstance(payload, dict):
            continue
        for actor_id, region_value in payload.items():
            if actor_id not in world.actors:
                continue
            region_id = _coerce_region_id(region_value)
            if region_id:
                mapping[str(actor_id)] = region_id

    for actor_id, actor in world.actors.items():
        if actor_id in mapping:
            continue
        region_id = _metadata_region_id(actor.metadata)
        if region_id:
            mapping[actor_id] = region_id
    return mapping


def _as_targets(value: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        targets: list[tuple[str, dict[str, Any]]] = []
        for target_id, target_meta in value.items():
            metadata = dict(target_meta) if isinstance(target_meta, dict) else {"weight": target_meta}
            targets.append((str(target_id), metadata))
        return targets
    if isinstance(value, str):
        return [(value, {})]
    if isinstance(value, Sequence):
        targets = []
        for item in value:
            if isinstance(item, dict):
                target = (
                    item.get("target")
                    or item.get("target_region")
                    or item.get("target_geo_id")
                    or item.get("to")
                )
                if target is not None:
                    targets.append((str(target), dict(item)))
            else:
                targets.append((str(item), {}))
        return targets
    return []


def _spatial_adjacency_edges(spatial: dict[str, Any]) -> list[dict[str, Any]]:
    adjacency = spatial.get("adjacency", spatial.get("neighbors", spatial.get("spatial_adjacency", [])))
    if isinstance(adjacency, dict):
        edges: list[dict[str, Any]] = []
        for source_id, targets in adjacency.items():
            for target_id, metadata in _as_targets(targets):
                edges.append({"source": str(source_id), "target": target_id, **metadata})
        return edges
    if isinstance(adjacency, Sequence) and not isinstance(adjacency, str):
        edges = []
        for item in adjacency:
            if isinstance(item, dict):
                source = (
                    item.get("source")
                    or item.get("source_region")
                    or item.get("source_geo_id")
                    or item.get("from")
                )
                target = (
                    item.get("target")
                    or item.get("target_region")
                    or item.get("target_geo_id")
                    or item.get("to")
                )
                if source is not None and target is not None:
                    edges.append({"source": str(source), "target": str(target), **dict(item)})
            elif isinstance(item, Sequence) and not isinstance(item, str) and len(item) >= 2:
                edge = {"source": str(item[0]), "target": str(item[1])}
                if len(item) >= 3:
                    edge["weight"] = item[2]
                edges.append(edge)
        return edges
    return []


def _spatial_weight(edge: dict[str, Any]) -> float:
    for key in ("spatial_weight", "adjacency", "weight"):
        if key in edge:
            try:
                return float(edge[key])
            except (TypeError, ValueError):
                return 1.0
    distance = edge.get("distance", edge.get("distance_km"))
    if distance is not None:
        try:
            return 1.0 / (1.0 + max(float(distance), 0.0))
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _spatial_relation_exists(world: WorldState, source_id: str, target_id: str, relation_type: str) -> bool:
    return any(
        relation.source_id == source_id
        and relation.target_id == target_id
        and relation.relation_type == relation_type
        for relation in world.relations
    )


def _estimated_relation_count(
    adjacency_edges: Sequence[dict[str, Any]],
    actors_by_region: dict[str, list[str]],
    *,
    include_inverse_default: bool,
    default_directed: bool,
) -> int:
    estimate = 0
    for edge in adjacency_edges:
        source_region = _coerce_region_id(edge.get("source"))
        target_region = _coerce_region_id(edge.get("target"))
        if not source_region or not target_region or source_region == target_region:
            continue
        directed = bool(edge.get("directed", default_directed))
        include_inverse = bool(edge.get("include_inverse", include_inverse_default and not directed))
        multiplier = 2 if include_inverse else 1
        estimate += multiplier * len(actors_by_region.get(source_region, [])) * len(actors_by_region.get(target_region, []))
    return estimate


def _warn_if_large_spatial_materialization(
    world: WorldState,
    adjacency_edges: Sequence[dict[str, Any]],
    actors_by_region: dict[str, list[str]],
    *,
    include_inverse_default: bool,
    default_directed: bool,
) -> None:
    if not adjacency_edges:
        return
    region_count = len(actors_by_region)
    estimated_relations = _estimated_relation_count(
        adjacency_edges,
        actors_by_region,
        include_inverse_default=include_inverse_default,
        default_directed=default_directed,
    )
    if region_count <= SPATIAL_REGION_WARNING_THRESHOLD and estimated_relations <= SPATIAL_RELATION_WARNING_THRESHOLD:
        return
    LOGGER.warning(
        "large_spatial_materialization domain_id=%s regions=%d adjacency_edges=%d estimated_relations=%d "
        "threshold_regions=%d threshold_relations=%d",
        world.domain_id,
        region_count,
        len(adjacency_edges),
        estimated_relations,
        SPATIAL_REGION_WARNING_THRESHOLD,
        SPATIAL_RELATION_WARNING_THRESHOLD,
    )


def initialize_spatial_relations(world: WorldState) -> int:
    """Materialize ``world.metadata['spatial'].adjacency`` as actor relations.

    Expected metadata:
    - ``spatial.adjacency`` as ``{"REG_A": ["REG_B"]}``, edge dicts, or ``[["REG_A", "REG_B", weight]]``.
    - ``spatial.actor_regions`` mapping actor ids to region ids, or actor metadata with ``geo_id``/``region_id``.

    Undirected adjacency adds both directions by default. Set ``spatial.directed=true`` or edge-level
    ``directed=true`` to emit only the declared direction.
    """

    spatial = _spatial_metadata(world)
    if not spatial:
        return 0

    actor_regions = _actor_region_map(world, spatial)
    if not actor_regions:
        return 0
    actors_by_region: dict[str, list[str]] = {}
    for actor_id, region_id in actor_regions.items():
        actors_by_region.setdefault(region_id, []).append(actor_id)

    default_relation_type = str(spatial.get("relation_type", SPATIAL_NEIGHBOR_RELATION))
    default_directed = bool(spatial.get("directed", False))
    include_inverse_default = bool(spatial.get("include_inverse", not default_directed))
    adjacency_edges = _spatial_adjacency_edges(spatial)
    _warn_if_large_spatial_materialization(
        world,
        adjacency_edges,
        actors_by_region,
        include_inverse_default=include_inverse_default,
        default_directed=default_directed,
    )
    added = 0
    materialized: list[dict[str, Any]] = []

    for edge in adjacency_edges:
        source_region = _coerce_region_id(edge.get("source"))
        target_region = _coerce_region_id(edge.get("target"))
        if not source_region or not target_region or source_region == target_region:
            continue
        relation_type = str(edge.get("relation_type", default_relation_type))
        weight = _spatial_weight(edge)
        distance = edge.get("distance", edge.get("distance_km"))
        directed = bool(edge.get("directed", default_directed))
        include_inverse = bool(edge.get("include_inverse", include_inverse_default and not directed))
        region_pairs = [(source_region, target_region)]
        if include_inverse:
            region_pairs.append((target_region, source_region))

        for left_region, right_region in region_pairs:
            for source_actor in actors_by_region.get(left_region, []):
                for target_actor in actors_by_region.get(right_region, []):
                    if source_actor == target_actor:
                        continue
                    if _spatial_relation_exists(world, source_actor, target_actor, relation_type):
                        continue
                    weights = {
                        "adjacency": float(weight),
                        "spatial_weight": float(weight),
                    }
                    if distance is not None:
                        try:
                            weights["distance"] = float(distance)
                        except (TypeError, ValueError):
                            pass
                    world.add_relation(
                        Relation(
                            source_id=source_actor,
                            target_id=target_actor,
                            relation_type=relation_type,
                            weights=weights,
                        )
                    )
                    materialized.append(
                        {
                            "source_actor": source_actor,
                            "target_actor": target_actor,
                            "source_region": left_region,
                            "target_region": right_region,
                            "relation_type": relation_type,
                            "weight": float(weight),
                        }
                    )
                    added += 1

    if materialized:
        trace = world.metadata.setdefault("_spatial_materialization", {})
        trace["relation_count"] = int(trace.get("relation_count", 0)) + added
        trace.setdefault("relations", []).extend(materialized)
    return added


__all__ = [
    "SPATIAL_NEIGHBOR_RELATION",
    "SPATIAL_REGION_WARNING_THRESHOLD",
    "SPATIAL_RELATION_WARNING_THRESHOLD",
    "initialize_spatial_relations",
]
