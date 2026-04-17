"""Tests for the document-driven Freeman librarian fork."""

from __future__ import annotations

import json

from freeman_librarian.agent.document_signalingestion import DocumentSignalIngestionEngine
from freeman_librarian.bootstrap import DocumentBootstrapper
from freeman_librarian.extractor import DocumentExtractor
from freeman_librarian.memory.knowledgegraph import KnowledgeGraph


def test_document_extractor_parses_entities_and_relations(tmp_path) -> None:
    source = tmp_path / "regulation.md"
    source.write_text(
        "\n".join(
            [
                "Department: Finance Department",
                "Role: Budget Owner",
                "Process: Budget Approval",
                "Finance Department owns Budget Approval.",
                "Budget Owner participates in Budget Approval.",
            ]
        ),
        encoding="utf-8",
    )

    extracted = DocumentExtractor().extract_path(source)

    entity_ids = {entity.id for entity in extracted.entities}
    relation_types = {relation.relation_type for relation in extracted.relations}

    assert "unit_finance_department" in entity_ids
    assert "role_budget_owner" in entity_ids
    assert "process_budget_approval" in entity_ids
    assert {"owns", "participates_in"} <= relation_types


def test_document_bootstrapper_builds_world_and_reconciles_to_kg(tmp_path) -> None:
    source = tmp_path / "ops_policy.md"
    source.write_text(
        "\n".join(
            [
                "Department: Operations",
                "Process: Incident Review",
                "Operations owns Incident Review.",
            ]
        ),
        encoding="utf-8",
    )
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)

    result = DocumentBootstrapper().bootstrap_paths([source], domain_id="ops_model", knowledge_graph=kg)

    assert result.world_state.domain_id == "ops_model"
    assert "unit_operations" in result.world_state.actors
    assert "process_incident_review" in result.world_state.resources
    assert result.verification_report.passed is True
    assert kg.get_node("document_ops_policy") is not None
    assert kg.get_node("unit:unit_operations") is not None


def test_document_signal_diff_detects_new_responsibility(tmp_path) -> None:
    old_doc = tmp_path / "policy_v1.md"
    new_doc = tmp_path / "policy_v2.md"
    old_doc.write_text("Department: Finance\nProcess: Budget Approval\n", encoding="utf-8")
    new_doc.write_text(
        "Department: Finance\nProcess: Budget Approval\nFinance owns Budget Approval.\n",
        encoding="utf-8",
    )

    extractor = DocumentExtractor()
    previous = extractor.extract_path(old_doc)
    current = extractor.extract_path(new_doc)
    changeset = DocumentSignalIngestionEngine().diff(previous, current)

    assert any(change.change_type == "new_responsibility" for change in changeset.changes)
    payload = DocumentSignalIngestionEngine().as_parameter_vector_payload(changeset)
    assert json.loads(json.dumps(payload))["change_counts"]["new_responsibility"] >= 1
