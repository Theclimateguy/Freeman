"""Domain template registry and multi-domain composition helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from freeman_librarian.core.multiworld import MultiDomainWorld, SharedResourceBus
from freeman_librarian.core.world import WorldState
from freeman_librarian.domain.compiler import DomainCompiler
from freeman_librarian.utils import deep_copy_jsonable


@dataclass
class DomainTemplate:
    """Reusable domain template stored as a schema payload."""

    template_id: str
    schema: Dict[str, Any]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "schema": deep_copy_jsonable(self.schema),
            "description": self.description,
            "tags": list(self.tags),
            "metadata": deep_copy_jsonable(self.metadata),
        }


class DomainTemplateRegistry:
    """Register templates, compile worlds, and build multi-domain worlds."""

    def __init__(self, compiler: DomainCompiler | None = None) -> None:
        self.compiler = compiler or DomainCompiler()
        self._templates: Dict[str, DomainTemplate] = {}

    def register(self, template: DomainTemplate) -> None:
        self._templates[template.template_id] = template

    def get(self, template_id: str) -> DomainTemplate:
        return self._templates[template_id]

    def list(self) -> List[str]:
        return sorted(self._templates)

    def compile_world(self, template_id: str) -> WorldState:
        template = self.get(template_id)
        return self.compiler.compile(deep_copy_jsonable(template.schema))

    def compile_many(self, template_ids: Iterable[str]) -> List[WorldState]:
        return [self.compile_world(template_id) for template_id in template_ids]

    def build_multiworld(self, template_ids: Iterable[str], shared_resource_ids: List[str]) -> MultiDomainWorld:
        return MultiDomainWorld(self.compile_many(template_ids), shared_resource_ids)


__all__ = ["DomainTemplate", "DomainTemplateRegistry", "MultiDomainWorld", "SharedResourceBus"]
