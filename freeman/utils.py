"""Shared numeric and JSON helpers."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

EPSILON = float(np.float64(1.0e-8))
SIGN_EPSILON = float(np.float64(1.0e-4))


def to_float64(value: Any) -> Any:
    """Convert numeric scalars to ``numpy.float64`` and leave other values unchanged."""

    if isinstance(value, np.generic):
        return np.float64(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return np.float64(value)
    return value


def normalize_numeric_tree(value: Any) -> Any:
    """Recursively convert numeric leaves in lists and dictionaries to float64."""

    if isinstance(value, dict):
        return {str(k): normalize_numeric_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_numeric_tree(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_numeric_tree(v) for v in value]
    return to_float64(value)


def json_ready(value: Any) -> Any:
    """Convert a Python object tree into a JSON-serializable structure."""

    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def deep_copy_jsonable(value: Any) -> Any:
    """Deep-copy a JSON-like object tree."""

    return copy.deepcopy(value)


def stable_json_dumps(value: Any) -> str:
    """Serialize a JSON-like object with deterministic key ordering."""

    return json.dumps(json_ready(value), sort_keys=True)
