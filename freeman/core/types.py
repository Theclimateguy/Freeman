"""Core dataclasses for Freeman."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np

from freeman.utils import deep_copy_jsonable, json_ready, normalize_numeric_tree, to_float64


def _decode_float(value: Any) -> Any:
    """Decode special float markers used in snapshots."""

    if value == "inf":
        return np.float64(np.inf)
    if value == "-inf":
        return np.float64(-np.inf)
    return to_float64(value)


def _encode_float(value: Any) -> Any:
    """Encode float values for strict JSON serialization."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
    return json_ready(value)


@dataclass
class Actor:
    """Domain actor with arbitrary numeric state and metadata."""

    id: str
    name: str
    state: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = {k: np.float64(v) for k, v in normalize_numeric_tree(self.state).items()}
        self.metadata = normalize_numeric_tree(self.metadata)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable actor snapshot."""

        return {
            "id": self.id,
            "name": self.name,
            "state": json_ready(self.state),
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "Actor":
        """Recreate an actor from a snapshot."""

        return cls(
            id=data["id"],
            name=data["name"],
            state={k: _decode_float(v) for k, v in data.get("state", {}).items()},
            metadata=deep_copy_jsonable(data.get("metadata", {})),
        )


@dataclass
class Resource:
    """Numeric world resource with configurable evolution operator."""

    id: str
    name: str
    value: float
    unit: str
    owner_id: Optional[str] = None
    min_value: float = 0.0
    max_value: float = float("inf")
    evolution_type: str = "stock_flow"
    evolution_params: Dict[str, Any] = field(default_factory=dict)
    conserved: bool = False

    def __post_init__(self) -> None:
        self.value = np.float64(self.value)
        self.min_value = np.float64(self.min_value)
        self.max_value = _decode_float(self.max_value)
        self.evolution_params = normalize_numeric_tree(self.evolution_params)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable resource snapshot."""

        return {
            "id": self.id,
            "name": self.name,
            "value": _encode_float(self.value),
            "unit": self.unit,
            "owner_id": self.owner_id,
            "min_value": _encode_float(self.min_value),
            "max_value": _encode_float(self.max_value),
            "evolution_type": self.evolution_type,
            "evolution_params": json_ready(self.evolution_params),
            "conserved": self.conserved,
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "Resource":
        """Recreate a resource from a snapshot."""

        return cls(
            id=data["id"],
            name=data["name"],
            value=_decode_float(data["value"]),
            unit=data["unit"],
            owner_id=data.get("owner_id"),
            min_value=_decode_float(data.get("min_value", 0.0)),
            max_value=_decode_float(data.get("max_value", "inf")),
            evolution_type=data.get("evolution_type", "stock_flow"),
            evolution_params=deep_copy_jsonable(data.get("evolution_params", {})),
            conserved=bool(data.get("conserved", False)),
        )


@dataclass
class Relation:
    """Directed typed relation between actors."""

    source_id: str
    target_id: str
    relation_type: str
    weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = {k: np.float64(v) for k, v in normalize_numeric_tree(self.weights).items()}

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable relation snapshot."""

        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weights": json_ready(self.weights),
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "Relation":
        """Recreate a relation from a snapshot."""

        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            weights={k: _decode_float(v) for k, v in data.get("weights", {}).items()},
        )


@dataclass
class Outcome:
    """Named outcome scored from world values."""

    id: str
    label: str
    scoring_weights: Dict[str, float]
    description: str = ""

    def __post_init__(self) -> None:
        self.scoring_weights = {
            k: np.float64(v) for k, v in normalize_numeric_tree(self.scoring_weights).items()
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable outcome snapshot."""

        return {
            "id": self.id,
            "label": self.label,
            "scoring_weights": json_ready(self.scoring_weights),
            "description": self.description,
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "Outcome":
        """Recreate an outcome from a snapshot."""

        return cls(
            id=data["id"],
            label=data["label"],
            scoring_weights={k: _decode_float(v) for k, v in data.get("scoring_weights", {}).items()},
            description=data.get("description", ""),
        )


@dataclass
class CausalEdge:
    """Expected qualitative causal relationship between two world quantities."""

    source: str
    target: str
    expected_sign: Literal["+", "-"]
    strength: Literal["strong", "weak"] = "strong"

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable edge snapshot."""

        return {
            "source": self.source,
            "target": self.target,
            "expected_sign": self.expected_sign,
            "strength": self.strength,
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "CausalEdge":
        """Recreate a causal edge from a snapshot."""

        return cls(
            source=data["source"],
            target=data["target"],
            expected_sign=data["expected_sign"],
            strength=data.get("strength", "strong"),
        )


@dataclass
class Policy:
    """Actor policy as a mapping from action names to intensities."""

    actor_id: str
    actions: Dict[str, float]

    def __post_init__(self) -> None:
        self.actions = {k: np.float64(v) for k, v in normalize_numeric_tree(self.actions).items()}

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable policy snapshot."""

        return {"actor_id": self.actor_id, "actions": json_ready(self.actions)}

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "Policy":
        """Recreate a policy from a snapshot."""

        return cls(
            actor_id=data["actor_id"],
            actions={k: _decode_float(v) for k, v in data.get("actions", {}).items()},
        )


@dataclass
class Violation:
    """Verifier violation emitted by one of the validation levels."""

    level: int
    check_name: str
    description: str
    severity: Literal["hard", "soft"]
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.level = int(self.level)
        self.details = normalize_numeric_tree(self.details)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable violation snapshot."""

        return {
            "level": self.level,
            "check_name": self.check_name,
            "description": self.description,
            "severity": self.severity,
            "details": json_ready(self.details),
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "Violation":
        """Recreate a violation from a snapshot."""

        return cls(
            level=data["level"],
            check_name=data["check_name"],
            description=data["description"],
            severity=data["severity"],
            details=deep_copy_jsonable(data.get("details", {})),
        )
