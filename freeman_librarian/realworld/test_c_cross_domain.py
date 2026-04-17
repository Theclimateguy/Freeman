"""Synthetic cross-domain causal reasoning test."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from freeman_librarian.llm.deepseek import DeepSeekChatClient
from freeman_librarian.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman_librarian.realworld.manifold import (
    _load_deepseek_api_key,
    freeman_probability_from_schema,
    freeman_probability_with_llm_signal,
)


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CrossDomainTarget:
    """One synthetic target domain in Test C."""

    domain_id: str
    question: str
    prior: float
    level: int
    expected_direction: str
    domain_polarity: str
    expected_mechanism: str
    mechanism_keywords: tuple[str, ...]


@dataclass(frozen=True)
class CrossDomainResult:
    """Observed posterior shift for one cross-domain target."""

    domain_id: str
    question: str
    level: int
    prior_probability: float
    posterior_probability: float
    delta_probability: float
    expected_direction: str
    observed_direction: str
    direction_correct: bool
    expected_mechanism: str
    mechanism_keywords_hit: list[str]
    llm_rationale: str
    llm_parameter_vector: dict[str, Any]


_LEVEL_THRESHOLDS = {1: 1.0, 2: 2.0 / 3.0, 3: 2.0 / 3.0}
_ANCHOR_EVENT = {
    "event": "california_wildfire_2024_record",
    "domain": "climate_disaster",
    "value": 1.0,
    "magnitude": "record_season",
    "date": "2024-10-01",
}
_TARGETS: tuple[CrossDomainTarget, ...] = (
    CrossDomainTarget(
        domain_id="CA_air_quality_2024",
        question="Will California experience severe statewide air-quality deterioration from wildfire smoke in late 2024?",
        prior=0.40,
        level=1,
        expected_direction="up",
        domain_polarity="negative",
        expected_mechanism="wildfire smoke directly worsens air quality",
        mechanism_keywords=("smoke", "particulate", "air quality", "pm2.5"),
    ),
    CrossDomainTarget(
        domain_id="US_wildfire_insurance_losses_2024",
        question="Will U.S. insured wildfire losses in 2024 exceed current baseline expectations?",
        prior=0.55,
        level=1,
        expected_direction="up",
        domain_polarity="negative",
        expected_mechanism="record fire season directly raises insured losses",
        mechanism_keywords=("insured losses", "claims", "property damage", "losses"),
    ),
    CrossDomainTarget(
        domain_id="CA_homeowners_insurance_availability",
        question="Will homeowners insurance remain broadly available in California through 2025?",
        prior=0.60,
        level=2,
        expected_direction="down",
        domain_polarity="positive",
        expected_mechanism="insurer retreat after losses reduces availability",
        mechanism_keywords=("insurer", "carrier", "availability", "coverage"),
    ),
    CrossDomainTarget(
        domain_id="CA_net_migration_negative_2025",
        question="Will California net domestic migration be more negative in 2025 than current baseline expectations?",
        prior=0.45,
        level=2,
        expected_direction="up",
        domain_polarity="negative",
        expected_mechanism="housing disruption and insurance stress push migration outflows",
        mechanism_keywords=("migration", "outflow", "housing", "insurance"),
    ),
    CrossDomainTarget(
        domain_id="US_reinsurance_costs_increase",
        question="Will U.S. property-catastrophe reinsurance costs increase materially over the next 12 months?",
        prior=0.50,
        level=2,
        expected_direction="up",
        domain_polarity="negative",
        expected_mechanism="larger wildfire losses tighten reinsurance pricing",
        mechanism_keywords=("reinsurance", "pricing", "catastrophe", "losses"),
    ),
    CrossDomainTarget(
        domain_id="CA_real_estate_price_decline_fire_zones",
        question="Will residential real-estate prices decline in California fire-prone zones over the next 12 months?",
        prior=0.50,
        level=3,
        expected_direction="up",
        domain_polarity="negative",
        expected_mechanism="insurance withdrawal impairs transactions and prices",
        mechanism_keywords=("insurance", "property values", "housing market", "mortgage"),
    ),
    CrossDomainTarget(
        domain_id="municipal_bond_risk_CA_fire_counties",
        question="Will municipal bond risk increase in California fire-exposed counties over the next 12 months?",
        prior=0.40,
        level=3,
        expected_direction="up",
        domain_polarity="negative",
        expected_mechanism="weaker tax base and fiscal stress raise county credit risk",
        mechanism_keywords=("tax base", "county finances", "bond risk", "credit"),
    ),
    CrossDomainTarget(
        domain_id="federal_disaster_relief_supplemental_2025",
        question="Will Congress pass a supplemental federal disaster-relief package in 2025?",
        prior=0.35,
        level=3,
        expected_direction="up",
        domain_polarity="positive",
        expected_mechanism="larger disaster burden raises political pressure for federal relief",
        mechanism_keywords=("congress", "relief", "appropriation", "pressure"),
    ),
)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _logit(probability: float, eps: float = 1.0e-6) -> float:
    p = _clip(float(probability), eps, 1.0 - eps)
    return float(math.log(p / (1.0 - p)))


def _direction_label(delta: float, eps: float = 1.0e-6) -> str:
    if delta > eps:
        return "up"
    if delta < -eps:
        return "down"
    return "flat"


def build_cross_domain_schema(target: CrossDomainTarget) -> dict[str, Any]:
    """Build a generic binary schema from a target prior."""

    prior_logit = 0.5 * _logit(target.prior)
    return {
        "domain_id": target.domain_id,
        "name": target.question,
        "description": target.question,
        "domain_polarity": target.domain_polarity,
        "modifier_mode": "probability_monotonic",
        "actors": [
            {
                "id": "domain",
                "name": "Target Domain",
                "state": {"salience": 0.7},
                "metadata": {"level": target.level},
            }
        ],
        "resources": [
            {
                "id": "baseline_mass",
                "name": "Baseline Mass",
                "value": 1.0,
                "unit": "signal",
                "min_value": 0.0,
                "max_value": 4.0,
                "evolution_type": "linear",
                "evolution_params": {"a": 1.0, "b": 0.0, "c": 0.0, "coupling_weights": {}},
            },
            {
                "id": "prior_logit",
                "name": "Prior Logit",
                "value": prior_logit,
                "unit": "signal",
                "min_value": -4.0,
                "max_value": 4.0,
                "evolution_type": "linear",
                "evolution_params": {"a": 1.0, "b": 0.0, "c": 0.0, "coupling_weights": {}},
            }
        ],
        "relations": [],
        "outcomes": [
            {
                "id": "yes",
                "label": "YES",
                "scoring_weights": {"baseline_mass": 1.0, "prior_logit": 1.0},
                "description": target.question,
            },
            {
                "id": "no",
                "label": "NO",
                "scoring_weights": {"baseline_mass": 1.0, "prior_logit": -1.0},
                "description": f"NOT ({target.question})",
            },
        ],
        "causal_dag": [],
        "metadata": {
            "question": target.question,
            "prior_probability": target.prior,
            "level": target.level,
            "expected_mechanism": target.expected_mechanism,
            "domain_polarity": target.domain_polarity,
            "modifier_mode": "probability_monotonic",
        },
    }


def _signal_text(target: CrossDomainTarget) -> str:
    return "\n".join(
        [
            "ANCHOR EVENT JSON:",
            json.dumps(_ANCHOR_EVENT, ensure_ascii=False, sort_keys=True),
            "",
            "TARGET DOMAIN QUESTION:",
            f"- Domain ID: {target.domain_id}",
            f"- Literal YES question: {target.question}",
            f"- Prior YES probability: {target.prior:.2f}",
            "",
            "Instruction:",
            "Update this target domain only if the anchor event causally changes the literal YES probability.",
            "If you change the target, explain the intermediate mechanism explicitly.",
            "Do not use generic sentiment like 'wildfires are bad' without naming a transmission channel.",
        ]
    )


def _mechanism_keywords_hit(target: CrossDomainTarget, rationale: str) -> list[str]:
    lowered = rationale.lower()
    return [keyword for keyword in target.mechanism_keywords if keyword in lowered]


def _build_kg(output_dir: Path) -> KnowledgeGraph:
    return KnowledgeGraph(json_path=output_dir / "cross_domain_kg.json", auto_load=False, auto_save=False)


def _level_summary(results: Sequence[CrossDomainResult]) -> dict[int, dict[str, Any]]:
    summary: dict[int, dict[str, Any]] = {}
    for level in sorted({result.level for result in results}):
        level_results = [result for result in results if result.level == level]
        correct = sum(result.direction_correct for result in level_results)
        accuracy = float(correct / len(level_results))
        threshold = _LEVEL_THRESHOLDS[level]
        summary[level] = {
            "correct": int(correct),
            "total": len(level_results),
            "directional_accuracy": accuracy,
            "threshold": threshold,
            "passes_threshold": bool(accuracy >= threshold),
        }
    return summary


def run_cross_domain_causal_test(
    *,
    output_dir: str | Path,
    deepseek_api_key_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run synthetic Test C using one anchor event across three causal distances."""

    def emit(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    llm = DeepSeekChatClient(api_key=_load_deepseek_api_key(deepseek_api_key_path))
    kg = _build_kg(output_path)
    kg.add_node(
        KGNode(
            id=_ANCHOR_EVENT["event"],
            label="Anchor Event",
            node_type="event",
            content=json.dumps(_ANCHOR_EVENT, ensure_ascii=False, sort_keys=True),
            confidence=0.95,
            metadata=dict(_ANCHOR_EVENT),
        )
    )

    target_results: list[CrossDomainResult] = []
    level1_failed = False
    for target in _TARGETS:
        if level1_failed and target.level > 1:
            emit(f"[skip] {target.domain_id} skipped because level 1 sanity check failed.")
            continue
        schema = build_cross_domain_schema(target)
        prior_probability = freeman_probability_from_schema(schema)
        posterior_probability, parameter_vector = freeman_probability_with_llm_signal(
            schema,
            signal_text=_signal_text(target),
            llm_client=llm,
        )
        delta_probability = float(posterior_probability - prior_probability)
        observed_direction = _direction_label(delta_probability)
        direction_correct = bool(observed_direction == target.expected_direction)
        llm_rationale = str(parameter_vector.get("rationale", "")).strip()
        mechanism_hits = _mechanism_keywords_hit(target, llm_rationale)

        result = CrossDomainResult(
            domain_id=target.domain_id,
            question=target.question,
            level=target.level,
            prior_probability=float(prior_probability),
            posterior_probability=float(posterior_probability),
            delta_probability=delta_probability,
            expected_direction=target.expected_direction,
            observed_direction=observed_direction,
            direction_correct=direction_correct,
            expected_mechanism=target.expected_mechanism,
            mechanism_keywords_hit=mechanism_hits,
            llm_rationale=llm_rationale,
            llm_parameter_vector=parameter_vector,
        )
        target_results.append(result)

        kg.add_node(
            KGNode(
                id=target.domain_id,
                label=target.domain_id,
                node_type="domain",
                content=target.question,
                confidence=0.85,
                metadata={
                    "level": target.level,
                    "prior_probability": prior_probability,
                    "posterior_probability": posterior_probability,
                    "delta_probability": delta_probability,
                    "expected_direction": target.expected_direction,
                    "observed_direction": observed_direction,
                    "direction_correct": direction_correct,
                    "expected_mechanism": target.expected_mechanism,
                    "mechanism_keywords_hit": mechanism_hits,
                    "llm_rationale": llm_rationale,
                    "llm_parameter_vector": parameter_vector,
                },
            )
        )
        kg.add_edge(
            KGEdge(
                source=_ANCHOR_EVENT["event"],
                target=target.domain_id,
                relation_type=f"cross_domain_level_{target.level}",
                confidence=0.75,
                weight=float(max(0.4, 1.0 - 0.15 * (target.level - 1))),
                metadata={"expected_direction": target.expected_direction},
            )
        )
        emit(
            f"[L{target.level}] {target.domain_id} prior={prior_probability:.4f} "
            f"posterior={posterior_probability:.4f} delta={delta_probability:+.4f} "
            f"expected={target.expected_direction} observed={observed_direction} "
            f"mechanism_hits={len(mechanism_hits)}"
        )

        if target.level == 1:
            level1_results = [row for row in target_results if row.level == 1]
            if len(level1_results) == 2 and not all(row.direction_correct for row in level1_results):
                level1_failed = True

    kg_path = kg.save()
    levels = _level_summary(target_results)
    overall_pass = all(levels[level]["passes_threshold"] for level in levels)
    summary = {
        "anchor_event": dict(_ANCHOR_EVENT),
        "notes": [
            "Domains are evaluated separately; the LLM sees the anchor event and one target question at a time.",
            "No explicit expected direction or causal explanation is provided to the LLM.",
            "If level 1 fails, later levels are skipped by protocol.",
            f"DeepSeek DS.txt is used for every target-domain update via model={llm.model}.",
        ],
        "levels": levels,
        "overall_passes_threshold": bool(overall_pass),
        "level1_failed_stop": bool(level1_failed),
        "targets": [asdict(result) for result in target_results],
        "knowledge_graph_path": str(kg_path),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = [
    "CrossDomainResult",
    "CrossDomainTarget",
    "build_cross_domain_schema",
    "run_cross_domain_causal_test",
]
