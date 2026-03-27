"""Built-in domain profile registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class DomainRegistry:
    """Load built-in domain schemas packaged with Freeman."""

    def __init__(self) -> None:
        profiles_dir = Path(__file__).resolve().parent / "profiles"
        self._profiles = {path.stem: path for path in profiles_dir.glob("*.json")}

    def list_profiles(self) -> List[str]:
        """Return the list of bundled profile ids."""

        return sorted(self._profiles)

    def load_schema(self, profile_id: str) -> Dict[str, Any]:
        """Load a bundled schema by id."""

        path = self._profiles[profile_id]
        return json.loads(path.read_text(encoding="utf-8"))
