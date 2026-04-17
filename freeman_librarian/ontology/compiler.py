"""Compile the organizational ontology into a Freeman world schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freeman_librarian.ontology.schema import OrganizationOntology, slugify


def _actor_state(entity_kind: str) -> dict[str, float]:
    base = {"authority": 0.6, "capacity": 0.7, "load": 0.3}
    if entity_kind == "unit":
        base.update({"authority": 0.9, "capacity": 0.9})
    elif entity_kind == "role":
        base.update({"authority": 0.75, "capacity": 0.8})
    elif entity_kind == "employee":
        base.update({"authority": 0.5, "capacity": 0.65})
    return base


@dataclass
class OrganizationalDomainCompiler:
    """Translate organizational entities and relations into Freeman schema JSON."""

    process_default_value: float = 1.0
    compliance_default_value: float = 1.0

    def compile_schema(self, ontology: OrganizationOntology) -> dict[str, Any]:
        actors = []
        resources = []
        outcomes = []
        relations = []
        causal_dag = []

        owner_by_process: dict[str, str] = {}
        for relation in ontology.relations:
            if relation.relation_type == "owns":
                owner_by_process[relation.target_id] = relation.source_id

        for entity in ontology.actors.values():
            actors.append(
                {
                    "id": entity.id,
                    "name": entity.label,
                    "state": _actor_state(entity.kind),
                    "metadata": {
                        "kind": entity.kind,
                        "aliases": list(entity.aliases),
                        "sources": list(entity.sources),
                        **dict(entity.metadata),
                    },
                }
            )

        for entity in ontology.resources.values():
            if entity.kind == "process":
                value = self.process_default_value
                unit = "coverage"
                evolution_params = {"delta": 0.05, "phi_params": {"base_inflow": 0.05}}
            elif entity.kind == "document":
                value = 1.0
                unit = "active"
                evolution_params = {"delta": 0.10, "phi_params": {"base_inflow": 0.10}}
            elif entity.kind == "system":
                value = 1.0
                unit = "availability"
                evolution_params = {"delta": 0.08, "phi_params": {"base_inflow": 0.08}}
            else:
                value = 1.0
                unit = "inventory"
                evolution_params = {"delta": 0.05, "phi_params": {"base_inflow": 0.05}}

            resources.append(
                {
                    "id": entity.id,
                    "name": entity.label,
                    "value": value,
                    "unit": unit,
                    "owner_id": owner_by_process.get(entity.id),
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "evolution_type": "stock_flow",
                    "evolution_params": evolution_params,
                }
            )
            if entity.kind == "process":
                outcomes.append(
                    {
                        "id": f"process_status.{slugify(entity.id)}",
                        "label": f"Process status: {entity.label}",
                        "description": f"Operational coverage for process {entity.label}",
                        "scoring_weights": {entity.id: 1.0},
                    }
                )

        if not outcomes:
            outcomes.append(
                {
                    "id": "org_coverage",
                    "label": "Organization coverage",
                    "description": "Aggregate process coverage across the organizational corpus.",
                    "scoring_weights": {entity.id: 1.0 for entity in ontology.resources.values()},
                }
            )

        compliance_targets = [
            entity.id for entity in ontology.outcomes.values() if entity.kind == "compliance_state"
        ] or [entity.id for entity in ontology.resources.values() if entity.kind == "process"]
        outcomes.append(
            {
                "id": "org_compliance",
                "label": "Organizational compliance",
                "description": "Consistency and compliance score derived from documented processes.",
                "scoring_weights": {target_id: 1.0 for target_id in compliance_targets},
            }
        )

        actor_ids = {entity.id for entity in ontology.actors.values()}
        process_ids = {
            entity.id for entity in ontology.resources.values() if entity.kind == "process"
        }
        for relation in ontology.relations:
            if relation.relation_type in {"delegates_to", "reports_to"} and relation.source_id in actor_ids and relation.target_id in actor_ids:
                relations.append(
                    {
                        "source_id": relation.source_id,
                        "target_id": relation.target_id,
                        "relation_type": relation.relation_type,
                        "weights": {"authority": relation.confidence},
                    }
                )
            if relation.relation_type == "owns" and relation.source_id in actor_ids and relation.target_id in process_ids:
                causal_dag.append(
                    {
                        "source": f"{relation.source_id}.authority",
                        "target": relation.target_id,
                        "expected_sign": "+",
                        "strength": "strong",
                        "metadata": {
                            "relation_type": relation.relation_type,
                            "evidence": list(relation.evidence),
                        },
                    }
                )
            elif relation.relation_type == "participates_in" and relation.source_id in actor_ids and relation.target_id in process_ids:
                causal_dag.append(
                    {
                        "source": f"{relation.source_id}.capacity",
                        "target": relation.target_id,
                        "expected_sign": "+",
                        "strength": "weak",
                        "metadata": {
                            "relation_type": relation.relation_type,
                            "evidence": list(relation.evidence),
                        },
                    }
                )
            elif relation.relation_type == "requires" and relation.source_id in process_ids:
                causal_dag.append(
                    {
                        "source": relation.target_id,
                        "target": relation.source_id,
                        "expected_sign": "+",
                        "strength": "strong",
                        "metadata": {
                            "relation_type": relation.relation_type,
                            "evidence": list(relation.evidence),
                        },
                    }
                )

        return {
            "domain_id": ontology.domain_id,
            "name": ontology.metadata.get("name", ontology.domain_id),
            "description": ontology.metadata.get("description", "Document-derived organizational world model."),
            "actors": actors,
            "resources": resources,
            "relations": relations,
            "outcomes": outcomes,
            "causal_dag": causal_dag,
            "metadata": {
                "org": ontology.snapshot(),
                "document_ids": sorted(ontology.documents),
            },
        }


__all__ = ["OrganizationalDomainCompiler"]
