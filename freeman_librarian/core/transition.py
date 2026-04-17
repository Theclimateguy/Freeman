"""Main world transition operator."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from freeman_librarian.core.access import get_world_value
from freeman_librarian.core.evolution import effective_edge_weight, get_operator
from freeman_librarian.core.types import Policy, Violation
from freeman_librarian.core.world import WorldState
from freeman_librarian.exceptions import HardStopException
from freeman_librarian.verifier.level0 import level0_check


def _merge_policies(policies: List[Policy]) -> Dict[str, Policy]:
    """Aggregate multiple policies per actor into a single action map."""

    merged: Dict[str, Dict[str, np.float64]] = {}
    for policy in policies:
        merged.setdefault(policy.actor_id, {})
        for action, value in policy.actions.items():
            merged[policy.actor_id][action] = np.float64(merged[policy.actor_id].get(action, 0.0)) + np.float64(value)
    return {actor_id: Policy(actor_id=actor_id, actions=actions) for actor_id, actions in merged.items()}


def _update_actor_states(world: WorldState, next_world: WorldState, policy_map: Dict[str, Policy]) -> None:
    """Update actor state vectors from metadata-defined linear rules when present."""

    rules = world.actor_update_rules or world.metadata.get("actor_state_update", {})
    if not rules:
        return

    if all(actor_id in world.actors for actor_id in rules):
        actor_rule_map = rules
    else:
        actor_rule_map = {actor_id: rules for actor_id in world.actors}

    for actor_id, actor in world.actors.items():
        state_rules = actor_rule_map.get(actor_id, {})
        if not state_rules:
            continue
        next_actor = next_world.actors[actor_id]
        actor_policy = policy_map.get(actor_id)
        action_sum = np.sum(list(actor_policy.actions.values()), dtype=np.float64) if actor_policy else np.float64(0.0)
        for state_key, spec in state_rules.items():
            base = np.float64(spec.get("base", 0.0))
            decay = np.float64(spec.get("decay", 1.0))
            policy_scale = np.float64(spec.get("policy_scale", 0.0))
            value = decay * np.float64(actor.state.get(state_key, 0.0)) + base + policy_scale * action_sum
            for source_key, weight in spec.get("weights", {}).items():
                adjusted_weight = effective_edge_weight(world, str(source_key), state_key, weight)
                value += adjusted_weight * np.float64(get_world_value(next_world, source_key))
            if "min_value" in spec:
                value = max(value, np.float64(spec["min_value"]))
            if "max_value" in spec:
                value = min(value, np.float64(spec["max_value"]))
            next_actor.state[state_key] = np.float64(value)


def step_world(world: WorldState, policies: List[Policy], dt: float = 1.0) -> tuple[WorldState, List[Violation]]:
    """Advance the world by one timestep and run level-0 verification."""

    next_world = world.clone()
    violations: List[Violation] = []
    policy_map = _merge_policies(policies)

    for res_id, resource in world.resources.items():
        operator = get_operator(resource.evolution_type, resource.evolution_params)
        actor_policy = policy_map.get(resource.owner_id) if resource.owner_id else None
        new_value = operator.step(resource, world, actor_policy, dt)
        next_world.resources[res_id].value = np.float64(new_value)

    _update_actor_states(world, next_world, policy_map)

    level0_violations = level0_check(world, next_world)
    violations.extend(level0_violations)
    if any(violation.severity == "hard" for violation in level0_violations):
        raise HardStopException(level0_violations)

    next_world.t = world.t + 1
    next_world.seed = world.seed
    return next_world, violations
