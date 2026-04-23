"""Pytest collection rules for the Freeman lite branch."""

from __future__ import annotations

import os
from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # type: ignore[override]
    """Ignore legacy full-fat tests that target removed subsystems."""

    del config
    normalized = str(collection_path)
    if os.getenv("FREEMAN_LITE_INCLUDE_LEGACY_TESTS", "").strip():
        return False
    return f"{os.sep}tests{os.sep}" in normalized and f"{os.sep}tests_lite{os.sep}" not in normalized
