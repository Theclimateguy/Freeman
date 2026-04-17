"""Outcome scoring and confidence estimation."""

from __future__ import annotations

import ast
import re
from typing import Dict, Iterable, Mapping

import numpy as np

from freeman_librarian.core.access import get_world_value
from freeman_librarian.core.types import Outcome, Violation
from freeman_librarian.core.world import WorldState


def _normalize_identifier(name: str) -> str:
    """Return a compact lowercase alias for one world quantity."""

    return re.sub(r"[^a-zA-Z0-9]+", "", name).lower()


def _baseline_context(world: WorldState) -> dict[str, dict[str, float]]:
    """Return baseline values stored for regime-shift evaluation."""

    baseline = world.metadata.get("_baseline_state", {})
    return {
        "resources": dict(baseline.get("resources", {})) if isinstance(baseline, dict) else {},
        "actors": dict(baseline.get("actors", {})) if isinstance(baseline, dict) else {},
        "metadata": dict(baseline.get("metadata", {})) if isinstance(baseline, dict) else {},
    }


def _shock_context(world: WorldState) -> dict[str, dict[str, float]]:
    """Return accumulated shock/deviation values stored on the world."""

    shock_state = world.metadata.get("_shock_state", {})
    return {
        "resources": dict(shock_state.get("resources", {})) if isinstance(shock_state, dict) else {},
        "actors": dict(shock_state.get("actors", {})) if isinstance(shock_state, dict) else {},
        "metadata": dict(shock_state.get("metadata", {})) if isinstance(shock_state, dict) else {},
    }


def _register_condition_value(context: Dict[str, float], name: str, *, level: float, deviation: float) -> None:
    """Register deviation and absolute aliases for one quantity."""

    aliases = {
        name,
        name.lower(),
        _normalize_identifier(name),
    }
    for alias in aliases:
        if not alias:
            continue
        context.setdefault(alias, float(deviation))
        context.setdefault(f"level_{alias}", float(level))
        context.setdefault(f"abs_{alias}", float(level))


def _condition_context(world: WorldState) -> Dict[str, float]:
    """Build a safe numeric context for regime-shift conditions.

    Plain identifiers map to deviations from the stored baseline. Absolute levels are
    available via ``level_<name>`` or ``abs_<name>`` aliases.
    """

    context: Dict[str, float] = {}
    baseline = _baseline_context(world)
    shock_state = _shock_context(world)
    for resource_id, resource in world.resources.items():
        level = float(resource.value)
        base = float(baseline["resources"].get(resource_id, level))
        deviation = float(shock_state["resources"].get(resource_id, level - base))
        _register_condition_value(context, resource_id, level=level, deviation=deviation)
    for actor_id, actor in world.actors.items():
        actor_base = baseline["actors"].get(actor_id, {})
        actor_shock = shock_state["actors"].get(actor_id, {})
        for key, value in actor.state.items():
            level = float(value)
            base = float(actor_base.get(key, level))
            deviation = float(actor_shock.get(key, level - base))
            _register_condition_value(context, f"{actor_id}.{key}", level=level, deviation=deviation)
            _register_condition_value(context, f"{actor_id}_{key}", level=level, deviation=deviation)
            _register_condition_value(context, key, level=level, deviation=deviation)
    for key, value in world.metadata.items():
        if str(key).startswith("_") or not isinstance(value, (int, float, np.generic)):
            continue
        level = float(value)
        base = float(baseline["metadata"].get(key, level))
        deviation = float(shock_state["metadata"].get(key, level - base))
        _register_condition_value(context, key, level=level, deviation=deviation)
    return context


def _normalize_condition(condition: str) -> str:
    """Translate SQL-style boolean operators into Python syntax."""

    normalized = re.sub(r"\bAND\b", " and ", condition, flags=re.IGNORECASE)
    normalized = re.sub(r"\bOR\b", " or ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bNOT\b", " not ", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def _safe_eval(node: ast.AST, context: Mapping[str, float]) -> float | bool:
    """Safely evaluate a restricted boolean / numeric AST."""

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, context)
    if isinstance(node, ast.BoolOp):
        values = [_safe_eval(value, context) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(value) for value in values)
        if isinstance(node.op, ast.Or):
            return any(bool(value) for value in values)
        raise ValueError(f"Unsupported boolean operator: {ast.dump(node)}")
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand, context)
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        if isinstance(node.op, ast.USub):
            return -float(operand)
        if isinstance(node.op, ast.UAdd):
            return +float(operand)
        raise ValueError(f"Unsupported unary operator: {ast.dump(node)}")
    if isinstance(node, ast.BinOp):
        left = float(_safe_eval(node.left, context))
        right = float(_safe_eval(node.right, context))
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError(f"Unsupported binary operator: {ast.dump(node)}")
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left, context)
        for operator, comparator in zip(node.ops, node.comparators, strict=False):
            right = _safe_eval(comparator, context)
            if isinstance(operator, ast.Lt):
                passed = float(left) < float(right)
            elif isinstance(operator, ast.LtE):
                passed = float(left) <= float(right)
            elif isinstance(operator, ast.Gt):
                passed = float(left) > float(right)
            elif isinstance(operator, ast.GtE):
                passed = float(left) >= float(right)
            elif isinstance(operator, ast.Eq):
                passed = float(left) == float(right)
            elif isinstance(operator, ast.NotEq):
                passed = float(left) != float(right)
            else:
                raise ValueError(f"Unsupported comparison operator: {ast.dump(node)}")
            if not passed:
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise KeyError(f"Unknown regime-shift variable: {node.id}")
        return float(context[node.id])
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(f"Unsupported constant in regime-shift condition: {node.value!r}")
    raise ValueError(f"Unsupported regime-shift expression: {ast.dump(node)}")


def regime_shift_matches(world: WorldState, condition: str) -> bool:
    """Safely evaluate one regime-shift condition."""

    tree = ast.parse(_normalize_condition(condition), mode="eval")
    return bool(_safe_eval(tree, _condition_context(world)))


def _apply_regime_shifts(outcome: Outcome, world: WorldState, base_score: np.float64) -> np.float64:
    """Apply multiplicative regime-shift rules on top of a linear base score."""

    score = np.float64(base_score)
    for shift in outcome.regime_shifts:
        condition = str(shift.get("condition", "")).strip()
        multiplier = np.float64(shift.get("multiplier", 1.0))
        if not condition:
            continue
        if regime_shift_matches(world, condition):
            score *= multiplier
    return score


def pre_modifier_outcome_scores(world: WorldState) -> Dict[str, float]:
    """Return outcome scores after static regime shifts, before parameter-vector modifiers."""

    raw_scores: Dict[str, np.float64] = {}
    for outcome_id, outcome in world.outcomes.items():
        score = np.float64(0.0)
        for key, weight in outcome.scoring_weights.items():
            score += np.float64(weight) * np.float64(get_world_value(world, key))
        raw_scores[outcome_id] = _apply_regime_shifts(outcome, world, score)
    return {key: float(value) for key, value in raw_scores.items()}


def _apply_outcome_modifier(score: float, modifier: float, *, modifier_mode: str = "legacy") -> float:
    """Apply one outcome modifier.

    In ``probability_monotonic`` mode, ``modifier > 1`` always increases an
    outcome's softmax probability and ``modifier < 1`` always decreases it,
    even when the current raw score is negative.
    """

    score64 = np.float64(score)
    modifier64 = np.float64(modifier)
    if modifier_mode == "probability_monotonic":
        if score64 < 0:
            return float(score64 / max(modifier64, np.float64(1.0e-8)))
        return float(score64 * modifier64)
    return float(score64 * modifier64)


def scored_outcome_scores(world: WorldState) -> Dict[str, float]:
    """Return outcome scores after regime shifts and active parameter-vector modifiers."""

    scores = pre_modifier_outcome_scores(world)
    modifier_mode = str(world.metadata.get("modifier_mode", "legacy"))
    return {
        outcome_id: _apply_outcome_modifier(
            score,
            float(world.parameter_vector.outcome_modifiers.get(outcome_id, 1.0)),
            modifier_mode=modifier_mode,
        )
        for outcome_id, score in scores.items()
    }


def raw_outcome_scores(world: WorldState) -> Dict[str, float]:
    """Backward-compatible alias for post-modifier outcome scores."""

    return scored_outcome_scores(world)


def softmax_distribution(scores: Mapping[str, float]) -> Dict[str, float]:
    """Return a numerically stable softmax distribution."""

    if not scores:
        return {}

    max_score = np.float64(max(scores.values()))
    exp_scores = {key: np.exp(np.float64(value) - max_score) for key, value in scores.items()}
    total = np.sum(list(exp_scores.values()), dtype=np.float64)
    return {key: float(np.float64(value / total)) for key, value in exp_scores.items()}


def score_outcomes(world: WorldState) -> Dict[str, float]:
    """Score outcomes from world values and return a softmax distribution."""

    return softmax_distribution(scored_outcome_scores(world))


def compute_confidence(outcome_probs: Dict[str, float], violations: Iterable[Violation]) -> float:
    """Compute confidence from outcome concentration and soft-violation count."""

    if not outcome_probs:
        return 0.0

    probs = np.array(list(outcome_probs.values()), dtype=np.float64)
    entropy = -np.sum(probs * np.log(probs + np.float64(1.0e-10)), dtype=np.float64)
    entropy_max = np.log(np.float64(len(probs))) if len(probs) > 1 else np.float64(1.0)
    entropy_factor = np.float64(1.0) - entropy / entropy_max if entropy_max > 0 else np.float64(1.0)
    soft_violations = sum(1 for violation in violations if violation.severity == "soft")
    violation_penalty = max(0.0, 1.0 - 0.05 * soft_violations)
    return float(np.round(entropy_factor * np.float64(violation_penalty), 4))


__all__ = [
    "compute_confidence",
    "pre_modifier_outcome_scores",
    "raw_outcome_scores",
    "regime_shift_matches",
    "scored_outcome_scores",
    "score_outcomes",
    "softmax_distribution",
]
