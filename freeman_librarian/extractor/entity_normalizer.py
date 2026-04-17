"""Normalization utilities for document-extracted organizational entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from freeman_librarian.ontology.schema import (
    ACTOR_KINDS,
    RESOURCE_KINDS,
    OrganizationEntity,
    default_actor_id,
    default_resource_id,
    merge_sources,
    slugify,
)


ACTOR_HINTS = {
    "employee": {"employee", "staff", "person"},
    "role": {"role", "position", "title"},
    "unit": {"department", "division", "unit", "team", "office"},
}

RESOURCE_HINTS = {
    "process": {"process", "procedure", "workflow", "approval", "review"},
    "document": {"document", "policy", "regulation", "instruction", "guideline"},
    "artifact": {"artifact", "record", "report", "register", "form"},
    "system": {"system", "platform", "application", "service", "database"},
}


def infer_kind(label: str, explicit_kind: str | None = None) -> str:
    """Infer the entity kind from explicit metadata or lexical hints."""

    if explicit_kind:
        normalized = str(explicit_kind).strip().lower()
        if normalized in ACTOR_KINDS | RESOURCE_KINDS:
            return normalized
    lowered = str(label).strip().lower()
    for kind, hints in ACTOR_HINTS.items():
        if any(hint in lowered for hint in hints):
            return kind
    for kind, hints in RESOURCE_HINTS.items():
        if any(hint in lowered for hint in hints):
            return kind
    return "process"


@dataclass
class EntityNormalizer:
    """Canonicalize ids, labels, and aliases for extracted entities."""

    def normalize_entity(
        self,
        label: str,
        *,
        kind: str | None = None,
        aliases: Iterable[str] = (),
        sources: Iterable[str] = (),
        confidence: float = 0.75,
        metadata: dict | None = None,
    ) -> OrganizationEntity:
        resolved_kind = infer_kind(label, kind)
        entity_id = (
            default_actor_id(label, resolved_kind)
            if resolved_kind in ACTOR_KINDS
            else default_resource_id(label, resolved_kind)
        )
        return OrganizationEntity(
            id=entity_id,
            label=str(label).strip(),
            kind=resolved_kind,
            aliases=sorted({slugify(alias).replace("_", " ") for alias in aliases if str(alias).strip()}),
            sources=merge_sources(sources),
            confidence=confidence,
            metadata=dict(metadata or {}),
        )


__all__ = ["EntityNormalizer", "infer_kind"]
