"""Evolution operators and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import numpy as np

from freeman.core.access import get_world_value
from freeman.core.types import Policy, Resource
from freeman.core.world import WorldState


def _policy_sum(policy: Optional[Policy]) -> np.float64:
    """Return the total action intensity of a policy."""

    if policy is None:
        return np.float64(0.0)
    return np.float64(np.sum(list(policy.actions.values()), dtype=np.float64))


def _coupling_term(world: WorldState, weights: Dict[str, Any]) -> np.float64:
    """Return a linear combination of world values."""

    total = np.float64(0.0)
    for key, weight in weights.items():
        total += np.float64(weight) * np.float64(get_world_value(world, key))
    return total


class EvolutionOperator(ABC):
    """Abstract interface for all resource evolution operators."""

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
        coupling_term = _coupling_term(world, self.coupling_weights)
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

    def _phi(self, world: WorldState, policy: Optional[Policy]) -> np.float64:
        base_inflow = np.float64(self.phi_params.get("base_inflow", 0.0))
        policy_scale = np.float64(self.phi_params.get("policy_scale", 0.0))
        coupling_weights = self.phi_params.get("coupling_weights", {})
        return base_inflow + policy_scale * _policy_sum(policy) + _coupling_term(world, coupling_weights)

    def step(
        self,
        resource: Resource,
        world: WorldState,
        policy: Optional[Policy],
        dt: float = 1.0,
    ) -> float:
        phi = self._phi(world, policy)
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
        forcing = self.external + self.policy_scale * _policy_sum(policy) + _coupling_term(world, self.coupling_weights)
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
        world: WorldState,
        policy: Optional[Policy],
        dt: float,
        branch: Dict[str, Any],
    ) -> np.float64:
        mode = branch.get("mode", "linear")
        action_sum = _policy_sum(policy)
        coupling = _coupling_term(world, branch.get("coupling_weights", {}))
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
        return float(np.float64(self._branch_value(current, world, policy, dt, branch)))

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


EVOLUTION_REGISTRY: Dict[str, Type[EvolutionOperator]] = {
    "linear": LinearTransition,
    "stock_flow": StockFlowTransition,
    "logistic": LogisticGrowthTransition,
    "threshold": ThresholdTransition,
    "coupled": CoupledTransition,
}


def get_operator(evolution_type: str, params: Dict[str, Any]) -> EvolutionOperator:
    """Instantiate an evolution operator from the registry."""

    operator_cls = EVOLUTION_REGISTRY[evolution_type]
    return operator_cls(**params)
