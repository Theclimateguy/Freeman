"""OpenAI-compatible tool functions for Freeman."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from freeman.core.transition import step_world
from freeman.core.types import Policy
from freeman.core.world import WorldState
from freeman.domain.compiler import DomainCompiler
from freeman.exceptions import HardStopException
from freeman.game.runner import GameRunner, SimConfig
from freeman.verifier.level1 import level1_check
from freeman.verifier.level2 import level2_check
from freeman.verifier.report import VerificationReport

WORLD_REGISTRY: Dict[str, WorldState] = {}
TRAJECTORY_REGISTRY: Dict[str, List[Dict[str, Any]]] = {}

FREEMAN_TOOLS = [
    {
        "name": "freeman_compile_domain",
        "description": "Compile a domain schema into a simulation world. Returns world_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "schema": {"type": "object", "description": "Domain schema JSON"},
            },
            "required": ["schema"],
        },
    },
    {
        "name": "freeman_run_simulation",
        "description": "Run simulation on a compiled world. Returns SimResult JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "world_id": {"type": "string"},
                "policies": {"type": "array"},
                "max_steps": {"type": "integer", "default": 50},
                "seed": {"type": "integer", "default": 42},
            },
            "required": ["world_id", "policies"],
        },
    },
    {
        "name": "freeman_get_world_state",
        "description": "Get current state of a world at timestep t.",
        "parameters": {
            "type": "object",
            "properties": {
                "world_id": {"type": "string"},
                "t": {"type": "integer", "default": -1},
            },
            "required": ["world_id"],
        },
    },
    {
        "name": "freeman_verify_domain",
        "description": "Run verification levels on a compiled world.",
        "parameters": {
            "type": "object",
            "properties": {
                "world_id": {"type": "string"},
                "levels": {"type": "array", "items": {"type": "integer"}, "default": [0, 1, 2]},
            },
            "required": ["world_id"],
        },
    },
]


def _next_world_id(domain_id: str) -> str:
    """Return a new in-memory world id."""

    return f"{domain_id}:{len(WORLD_REGISTRY) + 1}"


def _coerce_policies(policies: Iterable[Policy | Dict[str, Any]]) -> List[Policy]:
    """Convert policy-like inputs into ``Policy`` instances."""

    return [policy if isinstance(policy, Policy) else Policy.from_snapshot(policy) for policy in policies]


def freeman_compile_domain(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Compile ``schema`` into a world and store it in the in-memory registry."""

    compiler = DomainCompiler()
    world = compiler.compile(schema)
    world_id = _next_world_id(world.domain_id)
    WORLD_REGISTRY[world_id] = world
    TRAJECTORY_REGISTRY[world_id] = [world.snapshot()]
    return {"world_id": world_id, "validation_result": {"valid": True, "domain_id": world.domain_id}}


def freeman_run_simulation(
    world_id: str,
    policies: Iterable[Policy | Dict[str, Any]],
    max_steps: int = 50,
    seed: int = 42,
) -> str:
    """Run a simulation for a compiled world and return serialized JSON."""

    world = WORLD_REGISTRY[world_id].clone()
    world.seed = seed
    config = SimConfig(max_steps=max_steps, seed=seed)
    runner = GameRunner(config)
    result = runner.run(world, _coerce_policies(policies))
    TRAJECTORY_REGISTRY[world_id] = result.trajectory
    WORLD_REGISTRY[world_id] = WorldState.from_snapshot(result.trajectory[-1])
    return result.to_json()


def freeman_get_world_state(world_id: str, t: int = -1) -> Dict[str, Any]:
    """Return a stored world snapshot, defaulting to the latest timestep."""

    trajectory = TRAJECTORY_REGISTRY.get(world_id)
    if trajectory:
        index = len(trajectory) - 1 if t == -1 else t
        return trajectory[index]
    return WORLD_REGISTRY[world_id].snapshot()


def freeman_verify_domain(world_id: str, levels: Iterable[int] = (0, 1, 2)) -> Dict[str, Any]:
    """Run selected verification levels for a compiled world."""

    world = WORLD_REGISTRY[world_id].clone()
    requested_levels = list(levels)
    violations = []

    if 0 in requested_levels:
        try:
            _, level0_violations = step_world(world.clone(), [])
            violations.extend(level0_violations)
        except HardStopException as exc:
            violations.extend(exc.violations)

    if 1 in requested_levels:
        violations.extend(level1_check(world.clone(), SimConfig(seed=world.seed)))

    if 2 in requested_levels:
        violations.extend(
            level2_check(world.clone(), world.causal_dag, base_delta=SimConfig().level2_shock_delta)
        )

    report = VerificationReport(
        world_id=world_id,
        domain_id=world.domain_id,
        levels_run=requested_levels,
        violations=violations,
        passed=not any(violation.severity == "hard" for violation in violations),
        metadata={"violation_count": len(violations)},
    )
    return report.snapshot()
