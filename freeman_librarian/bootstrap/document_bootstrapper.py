"""Document-driven bootstrapper for the Freeman librarian fork."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from freeman_librarian.agent.document_signalingestion import DocumentChangeSet, DocumentSignalIngestionEngine
from freeman_librarian.core.world import WorldState
from freeman_librarian.domain.compiler import DomainCompiler
from freeman_librarian.extractor import DocumentExtractor, ExtractedDocument, EntityNormalizer
from freeman_librarian.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman_librarian.memory.reconciler import Reconciler, ReconciliationResult
from freeman_librarian.memory.sessionlog import KGDelta, SessionLog
from freeman_librarian.ontology import OrganizationOntology, OrganizationRelation
from freeman_librarian.ontology.compiler import OrganizationalDomainCompiler
from freeman_librarian.ontology.schema import OrganizationEntity
from freeman_librarian.verifier import Verifier


@dataclass
class DocumentBootstrapResult:
    """Artifacts produced by document bootstrap."""

    ontology: OrganizationOntology
    extracted_documents: list[ExtractedDocument]
    changesets: list[DocumentChangeSet]
    world_schema: dict[str, Any]
    world_state: WorldState
    verification_report: Any
    session_log: SessionLog
    reconciliation_result: ReconciliationResult


class DocumentBootstrapper:
    """Continuously build an organizational world model from a document corpus."""

    def __init__(
        self,
        *,
        extractor: DocumentExtractor | None = None,
        compiler: OrganizationalDomainCompiler | None = None,
        domain_compiler: DomainCompiler | None = None,
        verifier: Verifier | None = None,
        reconciler: Reconciler | None = None,
        signal_engine: DocumentSignalIngestionEngine | None = None,
    ) -> None:
        self.extractor = extractor or DocumentExtractor(normalizer=EntityNormalizer())
        self.compiler = compiler or OrganizationalDomainCompiler()
        self.domain_compiler = domain_compiler or DomainCompiler()
        self.verifier = verifier or Verifier()
        self.reconciler = reconciler or Reconciler()
        self.signal_engine = signal_engine or DocumentSignalIngestionEngine()

    def bootstrap_paths(
        self,
        paths: Iterable[str | Path],
        *,
        domain_id: str = "freeman_librarian",
        knowledge_graph: KnowledgeGraph | None = None,
        existing_world: WorldState | None = None,
    ) -> DocumentBootstrapResult:
        extracted = self.extractor.extract_paths(paths)
        ontology = self._build_ontology(extracted, domain_id=domain_id)
        world_schema = self.compiler.compile_schema(ontology)
        if existing_world is not None:
            world_schema = self._merge_existing_world_schema(world_schema, existing_world.snapshot())
        world_state = self.domain_compiler.compile(world_schema)
        verification_report = self.verifier.run(world_state, levels=(1, 2, 3))
        session_log = self._build_session_log(ontology, extracted, verification_report)
        kg = knowledge_graph or KnowledgeGraph(auto_load=False, auto_save=True)
        reconciliation_result = self.reconciler.reconcile(kg, session_log)
        changesets = self._changesets(extracted)
        world_state.metadata.setdefault("org", {})
        world_state.metadata["org"]["changesets"] = [
            {"document_id": changeset.document_id, "summary": changeset.summary()}
            for changeset in changesets
        ]
        return DocumentBootstrapResult(
            ontology=ontology,
            extracted_documents=extracted,
            changesets=changesets,
            world_schema=world_schema,
            world_state=world_state,
            verification_report=verification_report,
            session_log=session_log,
            reconciliation_result=reconciliation_result,
        )

    def _changesets(self, extracted: list[ExtractedDocument]) -> list[DocumentChangeSet]:
        by_logical_id: dict[str, list[ExtractedDocument]] = {}
        for item in extracted:
            logical_id = item.document.document_id
            by_logical_id.setdefault(logical_id, []).append(item)
        changesets: list[DocumentChangeSet] = []
        for versions in by_logical_id.values():
            ordered = sorted(versions, key=lambda item: item.document.text_hash)
            previous = None
            for current in ordered:
                changesets.append(self.signal_engine.diff(previous, current))
                previous = current
        return changesets

    def _build_ontology(self, extracted: list[ExtractedDocument], *, domain_id: str) -> OrganizationOntology:
        ontology = OrganizationOntology(
            domain_id=domain_id,
            metadata={"name": domain_id, "description": "World model compiled from organizational documents."},
        )
        referenced_actor_ids: set[str] = set()
        for item in extracted:
            ontology.add_document(item.document)
            ontology.add_entity(
                OrganizationEntity(
                    id=item.document.document_id,
                    label=item.document.title,
                    kind="document",
                    sources=[item.document.document_id],
                    confidence=1.0,
                    metadata={
                        "path": item.document.path,
                        "version": item.document.version,
                        "text_hash": item.document.text_hash,
                    },
                )
            )
            for entity in item.entities:
                ontology.add_entity(entity)
                if entity.kind in {"employee", "role", "unit"}:
                    referenced_actor_ids.add(entity.id)
            for relation in item.relations:
                ontology.add_relation(relation)
                ontology.add_relation(
                    OrganizationRelation(
                        source_id=item.document.document_id,
                        target_id=relation.target_id,
                        relation_type="mentions",
                        evidence=[item.document.document_id],
                        confidence=1.0,
                    )
                )
                if relation.target_id.startswith(("employee_", "role_", "unit_")):
                    referenced_actor_ids.add(relation.target_id)
                if relation.source_id.startswith(("employee_", "role_", "unit_")):
                    referenced_actor_ids.add(relation.source_id)

        ownership_map: dict[str, set[str]] = {}
        for relation in ontology.relations:
            if relation.relation_type != "owns":
                continue
            ownership_map.setdefault(relation.target_id, set()).add(relation.source_id)
        conflicts = []
        for process_id, owner_ids in sorted(ownership_map.items()):
            if len(owner_ids) <= 1:
                continue
            conflicts.append(
                {
                    "conflict_type": "multiple_process_owners",
                    "process_id": process_id,
                    "owner_ids": sorted(owner_ids),
                }
            )
        ontology.conflicts = conflicts
        ontology.metadata["referenced_actor_ids"] = sorted(referenced_actor_ids)
        ontology.metadata["conflict_count"] = len(conflicts)
        return ontology

    def _merge_existing_world_schema(self, new_schema: dict[str, Any], existing_snapshot: dict[str, Any]) -> dict[str, Any]:
        merged = dict(new_schema)
        for key in ("actors", "resources", "outcomes"):
            existing_items = {
                item["id"]: item
                for item in existing_snapshot.get(key, {}).values()
            } if isinstance(existing_snapshot.get(key), dict) else {item["id"]: item for item in existing_snapshot.get(key, [])}
            new_items = {item["id"]: item for item in merged.get(key, [])}
            existing_items.update(new_items)
            merged[key] = list(existing_items.values())

        for key in ("relations", "causal_dag"):
            existing_items = existing_snapshot.get(key, [])
            deduped: dict[str, dict[str, Any]] = {}
            for item in existing_items + merged.get(key, []):
                if key == "relations":
                    dedupe_key = f"{item['source_id']}::{item['target_id']}::{item['relation_type']}"
                else:
                    dedupe_key = f"{item['source']}::{item['target']}::{item['expected_sign']}"
                deduped[dedupe_key] = item
            merged[key] = list(deduped.values())
        return merged

    def _build_session_log(
        self,
        ontology: OrganizationOntology,
        extracted: list[ExtractedDocument],
        verification_report: Any,
    ) -> SessionLog:
        session = SessionLog(
            session_id=f"freeman_librarian:{uuid4().hex[:12]}",
            metadata={
                "domain_id": ontology.domain_id,
                "document_count": len(extracted),
                "verification_passed": bool(verification_report.passed),
            },
        )
        for document in ontology.documents.values():
            node = KGNode(
                id=document.document_id,
                label=document.title,
                node_type="document",
                content=document.path,
                confidence=1.0,
                sources=[document.path],
                metadata={"claim_key": document.document_id, **document.metadata},
            )
            session.add_kg_delta(KGDelta(operation="add_node", payload={"node": node.snapshot()}))

        for entity in [*ontology.actors.values(), *ontology.resources.values(), *ontology.outcomes.values()]:
            node_id = f"{entity.kind}:{entity.id}"
            node = KGNode(
                id=node_id,
                label=entity.label,
                node_type=entity.kind,
                content=f"{entity.kind}: {entity.label}",
                confidence=entity.confidence,
                evidence=list(entity.sources),
                sources=list(entity.sources),
                metadata={"claim_key": f"{entity.kind}:{entity.id}", "canonical_id": entity.id, **entity.metadata},
            )
            session.add_kg_delta(KGDelta(operation="add_node", payload={"node": node.snapshot()}))

        for relation in ontology.relations:
            edge = KGEdge(
                source=relation.source_id if relation.source_id.startswith("document_") else f"{self._node_prefix(relation.source_id)}:{relation.source_id}",
                target=relation.target_id if relation.target_id.startswith("document_") else f"{self._node_prefix(relation.target_id)}:{relation.target_id}",
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                metadata={"evidence": list(relation.evidence), **relation.metadata},
            )
            session.add_kg_delta(KGDelta(operation="add_edge", payload={"edge": edge.snapshot()}))

        for conflict in ontology.conflicts:
            process_id = conflict["process_id"]
            node = KGNode(
                id=f"conflict:{process_id}",
                label=f"Conflict: {process_id}",
                node_type="conflict",
                content=str(conflict),
                confidence=0.55,
                metadata={"claim_key": f"conflict:{process_id}", **conflict},
            )
            session.add_kg_delta(KGDelta(operation="add_node", payload={"node": node.snapshot()}, contradiction=1))

        for violation in verification_report.violations:
            violation_key_suffix = (
                violation.details.get("resource_id")
                or violation.details.get("field")
                or violation.details.get("owner_id")
                or violation.level
            )
            node = KGNode(
                id=f"violation:{violation.level}:{violation.check_name}:{uuid4().hex[:8]}",
                label=f"Verifier L{violation.level}: {violation.check_name}",
                node_type="verification",
                content=violation.description,
                confidence=0.9 if violation.severity == "hard" else 0.6,
                metadata={"claim_key": f"verification:{violation.check_name}:{violation_key_suffix}", **violation.snapshot()},
            )
            session.add_kg_delta(
                KGDelta(
                    operation="add_node",
                    payload={"node": node.snapshot()},
                    contradiction=1 if violation.severity == "hard" else 0,
                )
            )
        return session

    def _node_prefix(self, entity_id: str) -> str:
        return entity_id.split("_", 1)[0]

__all__ = ["DocumentBootstrapResult", "DocumentBootstrapper"]
