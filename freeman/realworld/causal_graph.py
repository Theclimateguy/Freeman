"""Causal graph consistency test over real Manifold backtest markets."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from freeman.llm.deepseek import DeepSeekChatClient
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.realworld.manifold import (
    MarketFeatures,
    _load_deepseek_api_key,
    build_binary_market_schema,
    freeman_probability_from_schema,
    freeman_probability_with_llm_signal,
)


@dataclass(frozen=True)
class TargetSpec:
    """One expected causal neighbor of the anchor event."""

    market_substring: str
    relation_type: str
    expected_direction: str
    edge_weight: float
    explanation: str


@dataclass(frozen=True)
class TargetResult:
    """Observed before/after update for one neighboring market."""

    market_id: str
    question: str
    cutoff_time: str
    target_probability: float
    before_probability: float
    after_probability: float
    delta_probability: float
    expected_direction: str
    observed_direction: str
    direction_correct: bool
    relation_type: str
    edge_weight: float
    llm_rationale: str
    llm_parameter_vector: dict[str, Any]


def _load_backtest_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).resolve()
    with source.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_row(rows: list[dict[str, str]], substring: str) -> dict[str, str]:
    needle = substring.lower()
    matches = [row for row in rows if needle in row["question"].lower()]
    if not matches:
        raise KeyError(f"No market matched substring: {substring}")
    if len(matches) > 1:
        raise ValueError(f"Multiple markets matched substring: {substring}")
    return matches[0]


def _row_to_features(row: dict[str, str]) -> MarketFeatures:
    return MarketFeatures(
        cutoff_probability=float(row["feature_cutoff_probability"]),
        probability_7d=float(row["feature_probability_7d"]),
        probability_30d=float(row["feature_probability_30d"]),
        momentum_7d=float(row["feature_momentum_7d"]),
        momentum_30d=float(row["feature_momentum_30d"]),
        flow_7d=float(row["feature_flow_7d"]),
        flow_30d=float(row["feature_flow_30d"]),
        turnover_7d=float(row["feature_turnover_7d"]),
        turnover_30d=float(row["feature_turnover_30d"]),
        bets_total=int(float(row["feature_bets_total"])),
        bets_7d=int(float(row["feature_bets_7d"])),
        bets_30d=int(float(row["feature_bets_30d"])),
        liquidity=float(row["feature_liquidity"]),
        age_days=float(row["feature_age_days"]),
        horizon_days=float(row["feature_horizon_days"]),
        cutoff_time_ms=int(float(row["feature_cutoff_time_ms"])),
    )


def _row_to_market(row: dict[str, str]) -> dict[str, Any]:
    return {
        "id": row["market_id"],
        "question": row["question"],
        "textDescription": row["question"],
        "createdTime": 0,
        "resolutionTime": 0,
        "totalLiquidity": float(row["feature_liquidity"]),
    }


def _signal_text(anchor_row: dict[str, str], target_row: dict[str, str], spec: TargetSpec) -> str:
    return "\n".join(
        [
            "OBSERVED ANCHOR EVENT:",
            f"- Market: {anchor_row['question']}",
            f"- Cutoff state in backtest: {anchor_row['cutoff_time']}",
            "- Realized event: YES / observed policy retreat",
            "",
            "TARGET DOMAIN:",
            f"- Market: {target_row['question']}",
            f"- Stored prior snapshot date: {target_row['cutoff_time']}",
            f"- Relation type: {spec.relation_type}",
            f"- Causal explanation: {spec.explanation}",
            "",
            "Instruction:",
            "Update the target domain only if the observed anchor event should causally change the literal YES probability of the target market.",
            "Reason about the literal market question, not about whether the world becomes better or worse in general.",
        ]
    )


def _direction_label(delta: float, eps: float = 1.0e-6) -> str:
    if delta > eps:
        return "up"
    if delta < -eps:
        return "down"
    return "flat"


def _build_kg(output_dir: Path) -> KnowledgeGraph:
    return KnowledgeGraph(json_path=output_dir / "causal_graph_kg.json", auto_load=False, auto_save=False)


def run_paris_causal_graph_test(
    *,
    backtest_csv: str | Path,
    output_dir: str | Path,
    deepseek_api_key_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the structural causal-graph test around Paris-withdrawal."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    rows = _load_backtest_rows(backtest_csv)

    anchor_row = _find_row(rows, "withdraw from the Paris Agreement a second time")
    target_specs = [
        TargetSpec(
            market_substring="Carbon Brief Forecast: If Biden wins, will annual US CO2 emissions be below 4.5 billion tons in 2030?",
            relation_type="policy_retreat_to_emissions_slippage",
            expected_direction="down",
            edge_weight=0.90,
            explanation="A Paris withdrawal weakens long-run decarbonization commitment, making the literal YES outcome less likely.",
        ),
        TargetSpec(
            market_substring="binding agreement to limit global warming to 1.5 deg C",
            relation_type="policy_retreat_to_cooperation_breakdown",
            expected_direction="down",
            edge_weight=0.95,
            explanation="A major US policy withdrawal should reduce the literal probability of a binding multilateral climate agreement.",
        ),
        TargetSpec(
            market_substring="deplete the remaining carbon budget to limit warming to 1.5",
            relation_type="policy_retreat_to_budget_depletion",
            expected_direction="up",
            edge_weight=0.85,
            explanation="A Paris withdrawal should increase the literal YES probability of exhausting the remaining 1.5C carbon budget.",
        ),
    ]

    llm = DeepSeekChatClient(api_key=_load_deepseek_api_key(deepseek_api_key_path))
    kg = _build_kg(output_path)
    kg.add_node(
        KGNode(
            id=anchor_row["market_id"],
            label="Anchor Event",
            node_type="domain",
            content=anchor_row["question"],
            confidence=0.95,
            metadata={
                "question": anchor_row["question"],
                "cutoff_time": anchor_row["cutoff_time"],
                "cutoff_probability": float(anchor_row["cutoff_probability"]),
                "target_probability": float(anchor_row["target_probability"]),
                "observed_event": "YES",
            },
        )
    )

    target_results: list[TargetResult] = []
    for spec in target_specs:
        target_row = _find_row(rows, spec.market_substring)
        market = _row_to_market(target_row)
        features = _row_to_features(target_row)
        schema = build_binary_market_schema(market, features)
        before_probability = freeman_probability_from_schema(schema)
        after_probability, parameter_vector = freeman_probability_with_llm_signal(
            schema,
            signal_text=_signal_text(anchor_row, target_row, spec),
            llm_client=llm,
        )
        delta_probability = float(after_probability - before_probability)
        observed_direction = _direction_label(delta_probability)
        direction_correct = observed_direction == spec.expected_direction
        llm_rationale = str(parameter_vector.get("rationale", "")).strip()

        kg.add_node(
            KGNode(
                id=target_row["market_id"],
                label=target_row["question"],
                node_type="domain",
                content=target_row["question"],
                confidence=0.85,
                metadata={
                    "question": target_row["question"],
                    "cutoff_time": target_row["cutoff_time"],
                    "target_probability": float(target_row["target_probability"]),
                    "before_probability": before_probability,
                    "after_probability": after_probability,
                    "delta_probability": delta_probability,
                    "expected_direction": spec.expected_direction,
                    "observed_direction": observed_direction,
                    "direction_correct": direction_correct,
                    "llm_rationale": llm_rationale,
                    "llm_parameter_vector": parameter_vector,
                },
            )
        )
        kg.add_edge(
            KGEdge(
                source=anchor_row["market_id"],
                target=target_row["market_id"],
                relation_type=spec.relation_type,
                confidence=0.8,
                weight=spec.edge_weight,
                metadata={
                    "expected_direction": spec.expected_direction,
                    "explanation": spec.explanation,
                },
            )
        )
        target_results.append(
            TargetResult(
                market_id=target_row["market_id"],
                question=target_row["question"],
                cutoff_time=target_row["cutoff_time"],
                target_probability=float(target_row["target_probability"]),
                before_probability=float(before_probability),
                after_probability=float(after_probability),
                delta_probability=delta_probability,
                expected_direction=spec.expected_direction,
                observed_direction=observed_direction,
                direction_correct=direction_correct,
                relation_type=spec.relation_type,
                edge_weight=spec.edge_weight,
                llm_rationale=llm_rationale,
                llm_parameter_vector=parameter_vector,
            )
        )

    kg_path = kg.save()
    passed = sum(result.direction_correct for result in target_results)
    summary = {
        "anchor_market_id": anchor_row["market_id"],
        "anchor_question": anchor_row["question"],
        "anchor_cutoff_time": anchor_row["cutoff_time"],
        "anchor_cutoff_probability": float(anchor_row["cutoff_probability"]),
        "anchor_target_probability": float(anchor_row["target_probability"]),
        "notes": [
            "This is a structural causal-consistency test over stored backtest priors, not a synchronous forecasting test.",
            "Target market cutoff dates differ from the anchor cutoff date; directions are evaluated on the literal YES semantics of each market question.",
            "DeepSeek DS.txt is used for every target-domain update via ParameterEstimator.",
        ],
        "targets": [asdict(result) for result in target_results],
        "correct_directions": int(passed),
        "target_count": len(target_results),
        "success_ratio": float(passed / len(target_results)),
        "passes_threshold": bool(passed / len(target_results) >= (2.0 / 3.0)),
        "knowledge_graph_path": str(kg_path),
    }
    (output_path / "summary.json").write_text(__import__("json").dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
