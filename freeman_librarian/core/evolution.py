"""Evolution operators and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Dict, Optional, Type

import numpy as np

from freeman_librarian.core.access import get_world_value
from freeman_librarian.core.types import Policy, Resource
from freeman_librarian.core.world import WorldState


def _policy_sum(policy: Optional[Policy]) -> np.float64:
    """Return the total action intensity of a policy."""

    if policy is None:
        return np.float64(0.0)
    return np.float64(np.sum(list(policy.actions.values()), dtype=np.float64))


def _normalize_edge_token(value: str) -> str:
    """Return a normalized token used for dynamic edge matching."""

    return re.sub(r"[^a-zA-Z0-9]+", "", value).lower()


def effective_edge_weight(world: WorldState, source_key: str, target_key: str, base_weight: Any) -> np.float64:
    """Return a coupling weight adjusted by the current parameter vector."""

    weight = np.float64(base_weight)
    parameter_vector = getattr(world, "parameter_vector", None)
    if parameter_vector is None:
        return weight

    normalized_source = _normalize_edge_token(source_key)
    normalized_target = _normalize_edge_token(target_key)
    for edge_key, delta in parameter_vector.edge_weight_deltas.items():
        if "." not in edge_key:
            continue
        edge_source, edge_target = edge_key.rsplit(".", 1)
        if _normalize_edge_token(edge_source) == normalized_source and _normalize_edge_token(edge_target) == normalized_target:
            weight += np.float64(delta)
    return weight


def _coupling_term(world: WorldState, weights: Dict[str, Any], *, target_key: str) -> np.float64:
    """Return a linear combination of world values."""

    total = np.float64(0.0)
    for key, weight in weights.items():
        total += effective_edge_weight(world, str(key), target_key, weight) * np.float64(get_world_value(world, key))
    return total


class EvolutionOperator(ABC):
    """Abstract interface for all resource evolution operators."""

    def delta(
        self,
        resource: Resource,
        world: WorldState,
        policy: Optional[Policy],
        dt: float = 1.0,
    ) -> float:
        """Return the net increment F(D, S(t), t) implied by the operator."""

        next_value = np.float64(self.step(resource, world, policy, dt))
        return float(next_value - np.float64(resource.value))

    @abstractmethod
    def step(
        self,
        resource: Resource,
        world: WorldState,
        policy: Optional[Policy],
        dt: float = 1.0,
    ) -> float:
        """Return the next resource value after one simulation step."""

    @abstractmethod
    def stability_bound(self) -> float:
        """Return a rough one-step growth bound used by structural verification."""


class LinearTransition(EvolutionOperator):
    """Affine resource transition with optional policy and coupling terms."""

    def __init__(
        self,
        a: float = 0.9,
        b: float = 0.0,
        c: float = 0.0,
        coupling_weights: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.a = np.float64(a)
        self.b = np.float64(b)
        self.c = np.float64(c)
        self.coupling_weights = coupling_weights or {}

    def step(
        self,
        resource: Resource,
        world: WorldState,
        policy: Optional[Policy],
        dt: float = 1.0,
    ) -> float:
        action_term = self.b * _policy_sum(policy)
        coupling_term = _coupling_term(world, self.coupling_weights, target_key=resource.id)
        value = self.a * np.float64(resource.value) + np.float64(dt) * (action_term + coupling_term + self.c)
        return float(np.float64(value))

    def stability_bound(self) -> float:
        """Return an affine upper bound based on operator coefficients."""

        return float(abs(self.a) + abs(self.b) + abs(self.c) + sum(abs(float(v)) for v in self.coupling_weights.values()))


class StockFlowTransition(EvolutionOperator):
    """Stock-flow operator with linear inflow specification."""

    def __init__(self, delta: float = 0.05, phi_params: Optional[Dict[str, Any]] = None) -> None:
        self.delta = np.float64(delta)
        self.phi_params = phi_params or {}

    def _phi(self, resource: Resource, world: WorldState, policy: Optional[Policy]) -> np.float64:
        base_inflow = np.float64(self.phi_params.get("base_inflow", 0.0))
        policy_scale = np.float64(self.phi_params.get("policy_scale", 0.0))
        coupling_weights = self.phi_params.get("coupling_weights", {})
        return base_inflow + policy_scale * _policy_sum(policy) + _coupling_term(world, coupling_weights, target_key=resource.id)

    def step(
        self,
        resource: Resource,
        world: WorldState,
        policy: Optional[Policy],
        dt: float = 1.0,
    ) -> float:
        phi = self._phi(resource, world, policy)
        value = np.float64(resource.value) + np.float64(dt) * (phi - self.delta * np.float64(resource.value))
        return float(np.float64(value))

    def stability_bound(self) -> float:
        """Return a simple bound derived from decay and inflow magnitudes."""

        coupling_bound = sum(abs(float(v)) for v in self.phi_params.get("coupling_weights", {}).values())
        return float(1.0 - self.delta + abs(float(self.phi_params.get("base_inflow", 0.0))) + coupling_bound)


class LogisticGrowthTransition(EvolutionOperator):
    """Logistic growth with optional external forcing."""

    def __init__(
        self,
        r: float = 0.1,
        K: float = 1.0,
        external: float = 0.0,
        policy_scale: float = 0.0,
        coupling_weights: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.r = np.float64(r)
        self.K = np.float64(K)
        self.external = np.float64(external)
        self.policy_scale = np.float64(policy_scale)
        self.coupling_weights = coupling_weights or {}

    def step(
        self,
        resource: Resource,
        world: WorldState,
        policy: Optional[Policy],
        dt: float = 1.0,
    ) -> float:
        current = np.float64(resource.value)
        growth = self.r * current * (1.0 - current / self.K)
        forcing = self.external + self.policy_scale * _policy_sum(policy) + _coupling_term(world, self.coupling_weights, target_key=resource.id)
        value = current + np.float64(dt) * (growth + forcing)
        value = np.clip(value, resource.min_value, min(resource.max_value, self.K))
        return float(np.float64(value))

    def stability_bound(self) -> float:
        """Return the carrying capacity as the effective bound."""

        return float(self.K)


class ThresholdTransition(EvolutionOperator):
    """Piecewise transition with low and high regimes."""

    def __init__(
        self,
        theta: float,
        low_params: Optional[Dict[str, Any]] = None,
        high_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.theta = np.float64(theta)
        self.low_params = low_params or {}
        self.high_params = high_params or {}

    def _branch_value(
        self,
        current: np.float64,
        target_key: str,
        world: WorldState,
        policy: Optional[Policy],
        dt: float,
        branch: Dict[str, Any],
    ) -> np.float64:
        mode = branch.get("mode", "linear")
        action_sum = _policy_sum(policy)
        coupling = _coupling_term(world, branch.get("coupling_weights", {}), target_key=target_key)
        if mode == "increment":
            delta = np.float64(branch.get("delta", 0.0))
            policy_scale = np.float64(branch.get("policy_scale", 0.0))
            return current + np.float64(dt) * (delta + policy_scale * action_sum + coupling)
        if mode == "stock_flow":
            base_inflow = np.float64(branch.get("base_inflow", 0.0))
            decay = np.float64(branch.get("delta", 0.0))
            policy_scale = np.float64(branch.get("policy_scale", 0.0))
            return current + np.float64(dt) * (base_inflow + policy_scale * action_sum + coupling - decay * current)
        a = np.float64(branch.get("a", 1.0))
        b = np.float64(branch.get("b", 0.0))
        c = np.float64(branch.get("c", 0.0))
        return a * current + np.float64(dt) * (b * action_sum + c + coupling)

    def step(
        self,
        resource: Resource,
        world: WorldState,
        policy: Optional[Policy],
        dt: float = 1.0,
    ) -> float:
        current = np.float64(resource.value)
        branch = self.low_params if current < self.theta else self.high_params
        return float(np.float64(self._branch_value(current, resource.id, world, policy, dt, branch)))

    def stability_bound(self) -> float:
        """Return the larger of the branch-specific bounds."""

        def _branch_bound(branch: Dict[str, Any]) -> float:
            mode = branch.get("mode", "linear")
            if mode == "linear":
                return abs(float(branch.get("a", 1.0)))
            if mode == "stock_flow":
                return max(0.0, 1.0 - float(branch.get("delta", 0.0)))
            if mode == "increment":
                return 1.0
            return 1.0

        return float(max(_branch_bound(self.low_params), _branch_bound(self.high_params)))


class CoupledTransition(EvolutionOperator):
    """Weighted composition of other transition operators."""

    def __init__(self, components: list[Dict[str, Any]]) -> None:
        self.components = components

    def step(
        self,
        resource: Resource,
        world: WorldState,
        policy: Optional[Policy],
        dt: float = 1.0,
    ) -> float:
        total = np.float64(0.0)
        for component in self.components:
            weight = np.float64(component.get("weight", 1.0))
            operator = get_operator(component["evolution_type"], component.get("evolution_params", {}))
            total += weight * np.float64(operator.step(resource, world, policy, dt))
        return float(np.float64(total))

    def stability_bound(self) -> float:
        """Return the weighted sum of component stability bounds."""

        total = np.float64(0.0)
        for component in self.components:
            weight = np.float64(abs(component.get("weight", 1.0)))
            operator = get_operator(component["evolution_type"], component.get("evolution_params", {}))
            total += weight * np.float64(operator.stability_bound())
        return float(total)


@dataclass
class EvolutionRegistry:
    """Factory/registry for named evolution operators."""

    operators: Dict[str, Type[EvolutionOperator]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operators:
            self.operators = dict(EVOLUTION_REGISTRY)

    def register(self, evolution_type: str, operator_cls: Type[EvolutionOperator]) -> None:
        """Register a new operator class."""

        self.operators[evolution_type] = operator_cls

    def get(self, evolution_type: str) -> Type[EvolutionOperator]:
        """Return the operator class registered under ``evolution_type``."""

        if evolution_type not in self.operators:
            raise KeyError(f"Unknown evolution_type: {evolution_type}")
        return self.operators[evolution_type]

    def create(self, evolution_type: str, params: Optional[Dict[str, Any]] = None) -> EvolutionOperator:
        """Instantiate an operator from the registry."""

        operator_cls = self.get(evolution_type)
        return operator_cls(**(params or {}))

    def available(self) -> tuple[str, ...]:
        """Return registered operator names in deterministic order."""

        return tuple(sorted(self.operators))


EVOLUTION_REGISTRY: Dict[str, Type[EvolutionOperator]] = {
    "linear": LinearTransition,
    "stock_flow": StockFlowTransition,
    "logistic": LogisticGrowthTransition,
    "threshold": ThresholdTransition,
    "coupled": CoupledTransition,
}
DEFAULT_EVOLUTION_REGISTRY = EvolutionRegistry(operators=EVOLUTION_REGISTRY)


def get_operator(evolution_type: str, params: Dict[str, Any]) -> EvolutionOperator:
    """Instantiate an evolution operator from the registry."""

    return DEFAULT_EVOLUTION_REGISTRY.create(evolution_type, params)


__all__ = [
    "CoupledTransition",
    "DEFAULT_EVOLUTION_REGISTRY",
    "EVOLUTION_REGISTRY",
    "EvolutionOperator",
    "EvolutionRegistry",
    "LinearTransition",
    "LogisticGrowthTransition",
    "StockFlowTransition",
    "ThresholdTransition",
    "get_operator",
]
