"""Unit tests for evolution operators."""

from __future__ import annotations

from freeman.core.evolution import (
    LinearTransition,
    LogisticGrowthTransition,
    StockFlowTransition,
    ThresholdTransition,
)
from freeman.core.types import Resource
from freeman.core.world import WorldState


def _empty_world() -> WorldState:
    return WorldState(
        domain_id="ops",
        t=0,
        actors={},
        resources={},
        relations=[],
        outcomes={},
        causal_dag=[],
    )


def test_stock_flow_stays_nonnegative() -> None:
    operator = StockFlowTransition(delta=0.2, phi_params={"base_inflow": 1.0})
    resource = Resource(id="r", name="R", value=10.0, unit="u", min_value=0.0)

    next_value = operator.step(resource, _empty_world(), None)

    assert next_value >= 0.0


def test_logistic_growth_is_bounded_by_capacity() -> None:
    operator = LogisticGrowthTransition(r=0.5, K=100.0, external=50.0)
    resource = Resource(id="r", name="R", value=99.0, unit="u", min_value=0.0, max_value=200.0)

    next_value = operator.step(resource, _empty_world(), None)

    assert next_value <= 100.0


def test_linear_transition_is_stable_for_a_less_than_one() -> None:
    operator = LinearTransition(a=0.8, b=0.0, c=0.0)
    resource = Resource(id="r", name="R", value=100.0, unit="u")

    value = resource.value
    for _ in range(20):
        resource.value = operator.step(resource, _empty_world(), None)
        value = resource.value

    assert value < 2.0


def test_threshold_stability_bound_respects_branch_modes() -> None:
    operator = ThresholdTransition(
        theta=1.0,
        low_params={"mode": "stock_flow", "delta": 0.2},
        high_params={"mode": "increment", "delta": 0.5},
    )

    assert operator.stability_bound() == 1.0
