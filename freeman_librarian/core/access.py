"""Helpers for reading and mutating world values."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from freeman_librarian.core.world import WorldState


def get_world_value(world: WorldState, key: str) -> float:
    """Return a resource value or actor-state aggregate referenced by ``key``."""

    if key in world.resources:
        return float(world.resources[key].value)

    if "." in key:
        actor_id, attr = key.split(".", 1)
        actor = world.actors.get(actor_id)
        if actor is not None:
            if attr in actor.state:
                return float(actor.state[attr])
            if attr in actor.metadata and isinstance(actor.metadata[attr], (int, float, np.generic)):
                return float(actor.metadata[attr])

    actor_matches = [float(actor.state[key]) for actor in world.actors.values() if key in actor.state]
    if actor_matches:
        return float(np.sum(actor_matches, dtype=np.float64))

    if key in world.metadata and isinstance(world.metadata[key], (int, float, np.generic)):
        return float(world.metadata[key])

    raise KeyError(f"Unknown world key: {key}")


def set_world_value(world: WorldState, key: str, value: float) -> None:
    """Set a resource value or actor-state aggregate referenced by ``key``."""

    cast_value = np.float64(value)

    if key in world.resources:
        world.resources[key].value = cast_value
        return

    if "." in key:
        actor_id, attr = key.split(".", 1)
        actor = world.actors.get(actor_id)
        if actor is None:
            raise KeyError(f"Unknown actor key: {key}")
        if attr in actor.state:
            actor.state[attr] = cast_value
            return
        actor.metadata[attr] = cast_value
        return

    matches = [actor for actor in world.actors.values() if key in actor.state]
    if matches:
        per_actor = cast_value / np.float64(len(matches))
        for actor in matches:
            actor.state[key] = np.float64(per_actor)
        return

    if key in world.metadata:
        world.metadata[key] = cast_value
        return

    raise KeyError(f"Unknown world key: {key}")


def apply_delta(world: WorldState, key: str, delta: float) -> WorldState:
    """Add a small perturbation to a world quantity and return the same world."""

    if key in world.resources:
        world.resources[key].value = np.float64(world.resources[key].value + np.float64(delta))
        return world

    if "." in key:
        actor_id, attr = key.split(".", 1)
        actor = world.actors.get(actor_id)
        if actor is None:
            raise KeyError(f"Unknown actor key: {key}")
        if attr in actor.state:
            actor.state[attr] = np.float64(actor.state[attr] + np.float64(delta))
            return world
        actor.metadata[attr] = np.float64(actor.metadata.get(attr, 0.0) + np.float64(delta))
        return world

    matches = [actor for actor in world.actors.values() if key in actor.state]
    if matches:
        per_actor = np.float64(delta) / np.float64(len(matches))
        for actor in matches:
            actor.state[key] = np.float64(actor.state[key] + per_actor)
        return world

    if key in world.metadata and isinstance(world.metadata[key], (int, float, np.generic)):
        world.metadata[key] = np.float64(world.metadata[key] + np.float64(delta))
        return world

    raise KeyError(f"Unknown world key: {key}")


def resource_vector(world: WorldState) -> np.ndarray:
    """Return resource values in deterministic order as a float64 vector."""

    return np.array([world.resources[key].value for key in sorted(world.resources)], dtype=np.float64)


def numeric_state_map(obj: WorldState | Dict[str, Any]) -> Dict[str, float]:
    """Flatten a world or snapshot into a comparable numeric mapping."""

    state: Dict[str, float] = {}
    if isinstance(obj, WorldState):
        for res_id, resource in obj.resources.items():
            state[f"resource:{res_id}"] = float(resource.value)
        for actor_id, actor in obj.actors.items():
            for key, value in actor.state.items():
                state[f"actor:{actor_id}:{key}"] = float(value)
        return state

    for res_id, resource in obj.get("resources", {}).items():
        state[f"resource:{res_id}"] = float(resource["value"])
    for actor_id, actor in obj.get("actors", {}).items():
        for key, value in actor.get("state", {}).items():
            state[f"actor:{actor_id}:{key}"] = float(value)
    return state


def state_distance(prev: WorldState | Dict[str, Any], curr: WorldState | Dict[str, Any]) -> float:
    """Compute the max absolute difference between two world states."""

    prev_map = numeric_state_map(prev)
    curr_map = numeric_state_map(curr)
    keys = sorted(set(prev_map) | set(curr_map))
    if not keys:
        return 0.0
    diffs = [abs(curr_map.get(key, 0.0) - prev_map.get(key, 0.0)) for key in keys]
    return float(np.max(np.array(diffs, dtype=np.float64)))
