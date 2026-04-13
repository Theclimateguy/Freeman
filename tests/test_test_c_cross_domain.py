"""Tests for synthetic cross-domain Test C helpers."""

from __future__ import annotations

from freeman.realworld.manifold import freeman_probability_from_schema
from freeman.realworld.test_c_cross_domain import CrossDomainTarget, build_cross_domain_schema
from freeman.core.scorer import score_outcomes
from freeman.core.types import ParameterVector
from freeman.domain.compiler import DomainCompiler


def test_build_cross_domain_schema_preserves_prior_probability() -> None:
    target = CrossDomainTarget(
        domain_id="demo",
        question="Will demo domain resolve YES?",
        prior=0.37,
        level=2,
        expected_direction="up",
        domain_polarity="negative",
        expected_mechanism="demo mechanism",
        mechanism_keywords=("demo",),
    )

    schema = build_cross_domain_schema(target)
    probability = freeman_probability_from_schema(schema)

    assert probability == target.prior


def test_build_cross_domain_schema_sets_probability_monotonic_metadata() -> None:
    target = CrossDomainTarget(
        domain_id="demo2",
        question="Will another demo domain resolve YES?",
        prior=0.63,
        level=3,
        expected_direction="down",
        domain_polarity="positive",
        expected_mechanism="another mechanism",
        mechanism_keywords=("another",),
    )

    schema = build_cross_domain_schema(target)

    assert schema["modifier_mode"] == "probability_monotonic"
    assert schema["domain_polarity"] == "positive"
    assert schema["metadata"]["prior_probability"] == 0.63


def test_cross_domain_schema_keeps_half_prior_movable_under_outcome_modifiers() -> None:
    target = CrossDomainTarget(
        domain_id="demo3",
        question="Will demo half-prior domain resolve YES?",
        prior=0.5,
        level=3,
        expected_direction="up",
        domain_polarity="negative",
        expected_mechanism="movable midpoint",
        mechanism_keywords=("midpoint",),
    )

    schema = build_cross_domain_schema(target)
    world = DomainCompiler().compile(schema)
    base_probability = score_outcomes(world)["yes"]

    world.parameter_vector = ParameterVector(outcome_modifiers={"yes": 2.0})
    updated_probability = score_outcomes(world)["yes"]

    assert base_probability == 0.5
    assert updated_probability > base_probability
