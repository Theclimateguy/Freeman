"""Organizational ontology primitives for document-driven world building."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Iterable
import re


def slugify(value: str) -> str:
    """Return a stable identifier fragment for organizational entities."""

    raw = str(value).strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", raw)
    cleaned = lowered.strip("_")
    if cleaned:
        return cleaned
    digest = sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"u_{digest}"


ACTOR_KINDS = {"employee", "role", "unit"}
RESOURCE_KINDS = {"process", "document", "artifact", "system"}
RELATION_KINDS = {"owns", "participates_in", "requires", "delegates_to", "reports_to", "mentions"}
OUTCOME_KINDS = {"process_status", "compliance_state"}


@dataclass
class OrganizationEntity:
    """Canonical organizational entity extracted from documents."""

    id: str
    label: str
    kind: str
    aliases: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.75
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.label = str(self.label).strip() or self.id
        self.kind = str(self.kind).strip().lower()
        self.aliases = sorted({str(alias).strip() for alias in self.aliases if str(alias).strip()})
        self.sources = sorted({str(source).strip() for source in self.sources if str(source).strip()})
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        self.metadata = dict(self.metadata)

    @property
    def entity_type(self) -> str:
        """Return the high-level Freeman bucket for the entity."""

        if self.kind in ACTOR_KINDS:
            return "actor"
        if self.kind in RESOURCE_KINDS:
            return "resource"
        if self.kind in OUTCOME_KINDS:
            return "outcome"
        raise ValueError(f"Unsupported organization entity kind: {self.kind}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "aliases": list(self.aliases),
            "sources": list(self.sources),
            "confidence": float(self.confidence),
            "metadata": dict(self.metadata),
        }


@dataclass
class OrganizationRelation:
    """Typed directed edge between canonical organizational entities."""

    source_id: str
    target_id: str
    relation_type: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.75
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_id = str(self.source_id)
        self.target_id = str(self.target_id)
        self.relation_type = str(self.relation_type).strip().lower()
        self.evidence = sorted({str(item).strip() for item in self.evidence if str(item).strip()})
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        self.metadata = dict(self.metadata)

    def key(self) -> tuple[str, str, str]:
        return (self.source_id, self.target_id, self.relation_type)

    def snapshot(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "evidence": list(self.evidence),
            "confidence": float(self.confidence),
            "metadata": dict(self.metadata),
        }


@dataclass
class OrganizationDocument:
    """Source document tracked by the librarian fork."""

    document_id: str
    title: str
    path: str
    version: str = ""
    text_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.document_id = str(self.document_id)
        self.title = str(self.title).strip() or self.document_id
        self.path = str(self.path)
        self.version = str(self.version)
        self.text_hash = str(self.text_hash)
        self.metadata = dict(self.metadata)

    def snapshot(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "path": self.path,
            "version": self.version,
            "text_hash": self.text_hash,
            "metadata": dict(self.metadata),
        }


@dataclass
class OrganizationOntology:
    """Aggregated organizational state compiled from a corpus."""

    domain_id: str
    actors: dict[str, OrganizationEntity] = field(default_factory=dict)
    resources: dict[str, OrganizationEntity] = field(default_factory=dict)
    outcomes: dict[str, OrganizationEntity] = field(default_factory=dict)
    relations: list[OrganizationRelation] = field(default_factory=list)
    documents: dict[str, OrganizationDocument] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity: OrganizationEntity) -> None:
        target = self.actors if entity.entity_type == "actor" else self.resources if entity.entity_type == "resource" else self.outcomes
        existing = target.get(entity.id)
        if existing is None:
            target[entity.id] = entity
            return
        merged_aliases = sorted(set(existing.aliases) | set(entity.aliases) | {existing.label, entity.label})
        merged_sources = sorted(set(existing.sources) | set(entity.sources))
        merged_metadata = {**existing.metadata, **entity.metadata}
        target[entity.id] = OrganizationEntity(
            id=entity.id,
            label=existing.label if len(existing.label) >= len(entity.label) else entity.label,
            kind=entity.kind,
            aliases=merged_aliases,
            sources=merged_sources,
            confidence=max(existing.confidence, entity.confidence),
            metadata=merged_metadata,
        )

    def add_relation(self, relation: OrganizationRelation) -> None:
        for index, existing in enumerate(self.relations):
            if existing.key() != relation.key():
                continue
            self.relations[index] = OrganizationRelation(
                source_id=relation.source_id,
                target_id=relation.target_id,
                relation_type=relation.relation_type,
                evidence=sorted(set(existing.evidence) | set(relation.evidence)),
                confidence=max(existing.confidence, relation.confidence),
                metadata={**existing.metadata, **relation.metadata},
            )
            return
        self.relations.append(relation)

    def add_document(self, document: OrganizationDocument) -> None:
        self.documents[document.document_id] = document

    def snapshot(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "actors": {key: entity.snapshot() for key, entity in self.actors.items()},
            "resources": {key: entity.snapshot() for key, entity in self.resources.items()},
            "outcomes": {key: entity.snapshot() for key, entity in self.outcomes.items()},
            "relations": [relation.snapshot() for relation in self.relations],
            "documents": {key: document.snapshot() for key, document in self.documents.items()},
            "conflicts": list(self.conflicts),
            "metadata": dict(self.metadata),
        }


def default_actor_id(label: str, kind: str) -> str:
    return f"{kind}_{slugify(label)}"


def default_resource_id(label: str, kind: str) -> str:
    return f"{kind}_{slugify(label)}"


def merge_sources(*collections: Iterable[str]) -> list[str]:
    merged: set[str] = set()
    for collection in collections:
        merged.update(str(item).strip() for item in collection if str(item).strip())
    return sorted(merged)


__all__ = [
    "ACTOR_KINDS",
    "OUTCOME_KINDS",
    "RELATION_KINDS",
    "RESOURCE_KINDS",
    "OrganizationDocument",
    "OrganizationEntity",
    "OrganizationOntology",
    "OrganizationRelation",
    "default_actor_id",
    "default_resource_id",
    "merge_sources",
    "slugify",
]
