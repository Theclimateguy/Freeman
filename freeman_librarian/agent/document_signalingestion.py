"""Document-aware signal ingestion and entity-level diffing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from freeman_librarian.extractor.document_extractor import ExtractedDocument


@dataclass
class EntityChange:
    """One semantic change between two document versions."""

    change_type: str
    entity_id: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChangeSet:
    """Entity-level diff between document versions."""

    document_id: str
    changes: list[EntityChange] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for change in self.changes:
            counts[change.change_type] = counts.get(change.change_type, 0) + 1
        return counts


class DocumentSignalIngestionEngine:
    """Diff entity graphs instead of raw text when a new document version arrives."""

    def diff(self, previous: ExtractedDocument | None, current: ExtractedDocument) -> DocumentChangeSet:
        current_entities = {entity.id: entity.snapshot() for entity in current.entities}
        previous_entities = {} if previous is None else {entity.id: entity.snapshot() for entity in previous.entities}
        current_relations = {relation.key(): relation.snapshot() for relation in current.relations}
        previous_relations = {} if previous is None else {relation.key(): relation.snapshot() for relation in previous.relations}

        changes: list[EntityChange] = []
        for entity_id in sorted(current_entities.keys() - previous_entities.keys()):
            kind = str(current_entities[entity_id].get("kind", "entity"))
            change_type = {
                "process": "new_process",
                "role": "new_actor",
                "employee": "new_actor",
                "unit": "new_actor",
            }.get(kind, "new_entity")
            changes.append(EntityChange(change_type=change_type, entity_id=entity_id, after=current_entities[entity_id]))
        for entity_id in sorted(previous_entities.keys() - current_entities.keys()):
            changes.append(EntityChange(change_type="removed_entity", entity_id=entity_id, before=previous_entities[entity_id]))
        for entity_id in sorted(current_entities.keys() & previous_entities.keys()):
            if current_entities[entity_id] == previous_entities[entity_id]:
                continue
            kind = str(current_entities[entity_id].get("kind", "entity"))
            change_type = "process_changed" if kind == "process" else "entity_changed"
            changes.append(
                EntityChange(
                    change_type=change_type,
                    entity_id=entity_id,
                    before=previous_entities[entity_id],
                    after=current_entities[entity_id],
                )
            )

        for relation_key in sorted(current_relations.keys() - previous_relations.keys()):
            relation = current_relations[relation_key]
            change_type = "new_responsibility" if relation["relation_type"] == "owns" else "new_relation"
            changes.append(
                EntityChange(
                    change_type=change_type,
                    entity_id=f"{relation['source_id']}->{relation['target_id']}",
                    after=relation,
                )
            )
        for relation_key in sorted(previous_relations.keys() & current_relations.keys()):
            if previous_relations[relation_key] == current_relations[relation_key]:
                continue
            relation = current_relations[relation_key]
            changes.append(
                EntityChange(
                    change_type="relation_changed",
                    entity_id=f"{relation['source_id']}->{relation['target_id']}",
                    before=previous_relations[relation_key],
                    after=relation,
                )
            )

        return DocumentChangeSet(document_id=current.document.document_id, changes=changes)

    def as_parameter_vector_payload(self, changeset: DocumentChangeSet) -> dict[str, Any]:
        """Map document changes into a lightweight parameter-style payload."""

        return {
            "document_id": changeset.document_id,
            "change_counts": changeset.summary(),
            "change_types": [change.change_type for change in changeset.changes],
        }


__all__ = ["DocumentChangeSet", "DocumentSignalIngestionEngine", "EntityChange"]
