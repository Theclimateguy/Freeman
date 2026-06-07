"""Preflight stratification for Test A market selection."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


_WS_RE = re.compile(r"\s+")

_NOISE_TERMS = (
    "room-temperature",
    "ambient-pressure",
    "superconductor",
    "lk-99",
    "ufo",
    "alien",
    "taylor swift",
    "pedophile",
    "homosexual",
    "elon musk",
    "zuckerberg",
    "youtube",
    "youtuber",
    "wikipedia",
    "apple",
    "google",
    "nba",
    "fifa",
    "damian lillard",
    "miami heat",
    "heat miser",
    "fani willis",
)
_LOCAL_WEATHER_TERMS = (
    "in nyc",
    "in sf",
    "helsinki",
    "that day in december",
    "that day in november",
    "that day in january",
    "mar 28, 2024",
    "mar 26, 2024",
    "may 31 in sf",
    "john's air conditioner",
)
_LEGAL_TERMS = (
    "held v montana",
    "plaintiffs win",
    "lawsuit",
    "court",
)
_POLICY_ACTOR_TERMS = (
    "biden",
    "harris",
    "trump",
    "president",
    "elected",
    "election",
    "white house",
    "government",
    "minister",
    "executive order",
    "world bank",
    "imf",
    "ballot",
)
_POLICY_REGULATION_TERMS = (
    "paris agreement",
    "binding agreement",
    "climate agreement",
    "withdraw from",
    "drop out of",
    "carbon credit system",
    "carbon dividend",
    "klimageld",
    "climate disaster relief fund",
    "cop28",
    "un climate change conference",
    "conference result",
    "canceled/relocated/postponed",
)
_POLICY_SALIENCE_TERMS = (
    "discussed in all official",
    "mentioned at all in the 2024 presidential debates",
    "replace its hvac system with heat pumps",
    "resign as minister",
)
_SECTOR_TERMS = (
    "co2 emissions",
    "greenhouse gas emissions",
    "carbon budget",
    "heat pumps",
    "climate non-profits",
)
_DIRECT_EMISSIONS_TERMS = (
    "global greenhouse gas emissions",
    "global co2 emissions",
    "us total co2 emissions",
    "global total co2 emissions",
    "carbon dioxide in may",
    "carbon dioxide in april",
    "carbon dioxide in march",
    "carbon dioxide in february",
    "carbon budget",
)
_PHYSICAL_CLIMATE_TERMS = (
    "ocean surface temperature",
    "sst",
    "wildfire",
    "acres burn",
    "flood",
    "drought",
    "global warming",
    "temperature anomaly",
    "climate anomaly",
    "heatwave",
    "heat wave",
    "ground temperature",
    "sea ice",
    "berkeley earth",
)
_WEAK_SIGNAL_TERMS = (
    "climate-tech startup",
    "climate positive public license",
    "dedicated crypto funds reallocate to climate",
    "terrorist attack motivated by climate change",
    "500 million deaths",
    "player in the 2022 fifa world cup",
)


@dataclass(frozen=True)
class TestAMarketRow:
    """One stratified market row for Test A preflight."""

    rank: int
    market_id: str
    question: str
    stratum: str
    selection_role: str
    include_test_a: bool
    priority: int
    signal_score: float
    causal_chain: str
    rationale: str


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _signal_score(question: str) -> float:
    text = _normalize(question)
    score = 0.0
    if "global" in text:
        score += 2.0
    if _contains_any(text, ("berkeley earth", "ocean surface temperature", "sst", "sea ice")):
        score += 2.0
    if _contains_any(text, ("wildfire", "flood", "drought", "heat wave", "heatwave")):
        score += 2.0
    if _contains_any(text, ("co2 emissions", "greenhouse gas emissions", "carbon dioxide")):
        score += 1.5
    if _contains_any(text, ("california", "us ", "world bank", "imf", "paris agreement", "cop28")):
        score += 1.0
    if _contains_any(text, _LOCAL_WEATHER_TERMS):
        score -= 3.0
    if _contains_any(text, ("before july 15", "before october 1", "by end of 2023")):
        score -= 0.5
    return float(score)


def _question_family(question: str) -> str:
    text = _normalize(question)
    if "carbon dioxide in " in text:
        return "monthly_co2_ppm"
    if "california wildfire season" in text or "acres burn" in text or "wildfire acreage" in text:
        return "california_wildfire"
    if "global warming be 1.5" in text and "berkeley earth" in text:
        return "berkeley_earth_1p5c"
    if "global greenhouse gas emissions" in text:
        return "global_ghg_emissions"
    if "global co2 emissions" in text:
        return "global_co2_emissions"
    if "us total co2 emissions" in text:
        return "us_co2_emissions"
    return text


def classify_test_a_market(question: str) -> tuple[str, str, bool, int, str, str]:
    """Classify one backtest market into a Test A stratum."""

    text = _normalize(question)
    if _contains_any(text, _NOISE_TERMS):
        return (
            "exclude_noise",
            "exclude",
            False,
            90,
            "lexical climate overlap but semantically unrelated",
            "Noise term matched unrelated topic.",
        )
    if _contains_any(text, _LOCAL_WEATHER_TERMS):
        return (
            "exclude_local_weather",
            "exclude",
            False,
            80,
            "local weather trivia rather than macro climate process",
            "Single-city or single-day weather question is weak for archived news evaluation.",
        )
    if _contains_any(text, _WEAK_SIGNAL_TERMS):
        return (
            "exclude_weak_signal",
            "exclude",
            False,
            70,
            "climate-adjacent but weak archival news signal",
            "Question is too diffuse or off-target for clean news-conditioned backtesting.",
        )
    if _contains_any(text, _POLICY_SALIENCE_TERMS):
        return (
            "manual_review_policy_salience",
            "manual_review",
            False,
            40,
            "political discourse or symbolic action -> uncertain policy transmission",
            "Potentially informative but causal path to climate outcome is too weak for default inclusion.",
        )
    if _contains_any(text, _LEGAL_TERMS):
        return (
            "legal_to_policy",
            "core_non_obvious",
            True,
            10,
            "climate litigation -> policy constraint / regulatory pressure",
            "Legal outcome can transmit into regulatory expectations with delayed market reaction.",
        )

    has_actor = _contains_any(text, _POLICY_ACTOR_TERMS)
    has_regulation = _contains_any(text, _POLICY_REGULATION_TERMS)
    has_sector = _contains_any(text, _SECTOR_TERMS)
    has_direct_emissions = _contains_any(text, _DIRECT_EMISSIONS_TERMS)
    has_physical = _contains_any(text, _PHYSICAL_CLIMATE_TERMS)

    if (has_actor or has_regulation) and has_sector:
        return (
            "policy_to_sector",
            "core_non_obvious",
            True,
            10,
            "political event -> regulation / implementation -> sectoral emissions outcome",
            "Non-obvious chain suitable for testing whether news adds beyond market prior.",
        )
    if has_actor or has_regulation:
        return (
            "policy_to_policy_finance",
            "core_non_obvious",
            True,
            12,
            "political event -> climate regulation / international coordination / finance",
            "Institutional climate markets are the main non-obvious bucket for Test A.",
        )
    if has_direct_emissions:
        return (
            "direct_emissions",
            "control",
            True,
            20,
            "direct emissions or atmospheric carbon outcome",
            "Useful control bucket with stronger and more direct news-outcome linkage.",
        )
    if has_physical:
        return (
            "physical_climate",
            "control",
            True,
            30,
            "physical climate event or measurement",
            "Useful control bucket for high-salience physical climate signals.",
        )
    return (
        "exclude_misc",
        "exclude",
        False,
        95,
        "outside target design for Test A",
        "Question does not fit the intended causal strata.",
    )


def stratify_backtest_rows(rows: Sequence[dict[str, str]]) -> list[TestAMarketRow]:
    """Apply the Test A preflight classifier to backtest rows."""

    stratified: list[TestAMarketRow] = []
    for rank, row in enumerate(rows, start=1):
        stratum, role, include, priority, causal_chain, rationale = classify_test_a_market(row["question"])
        stratified.append(
            TestAMarketRow(
                rank=rank,
                market_id=row["market_id"],
                question=row["question"],
                stratum=stratum,
                selection_role=role,
                include_test_a=include,
                priority=priority,
                signal_score=_signal_score(row["question"]),
                causal_chain=causal_chain,
                rationale=rationale,
            )
        )
    return stratified


def build_recommended_inclusion_list(
    stratified_rows: Sequence[TestAMarketRow],
    *,
    physical_control_limit: int = 12,
) -> list[TestAMarketRow]:
    """Build a deterministic inclusion list for Test A."""

    core_rows = [row for row in stratified_rows if row.selection_role == "core_non_obvious" and row.include_test_a]
    direct_controls = [row for row in stratified_rows if row.stratum == "direct_emissions" and row.include_test_a]
    physical_controls = [
        row
        for row in stratified_rows
        if row.stratum == "physical_climate" and row.include_test_a and row.signal_score >= 1.5
    ]
    selected_direct_controls = _cap_by_family(
        sorted(direct_controls, key=lambda row: (-row.signal_score, row.rank)),
        family_caps={"monthly_co2_ppm": 2},
    )
    selected_physical_controls = _cap_by_family(
        sorted(physical_controls, key=lambda row: (-row.signal_score, row.rank)),
        family_caps={
            "california_wildfire": 4,
            "berkeley_earth_1p5c": 2,
        },
        limit=max(int(physical_control_limit), 0),
    )
    selected = core_rows + selected_direct_controls + selected_physical_controls
    deduped: dict[str, TestAMarketRow] = {row.market_id: row for row in selected}
    return sorted(deduped.values(), key=lambda row: (row.priority, -row.signal_score, row.rank))


def _cap_by_family(
    rows: Sequence[TestAMarketRow],
    *,
    family_caps: dict[str, int],
    limit: int | None = None,
) -> list[TestAMarketRow]:
    selected: list[TestAMarketRow] = []
    family_counts: dict[str, int] = {}
    for row in rows:
        family = _question_family(row.question)
        cap = family_caps.get(family)
        if cap is not None and family_counts.get(family, 0) >= cap:
            continue
        selected.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        if limit is not None and len(selected) >= limit:
            break
    return selected


def run_test_a_preflight(
    *,
    backtest_csv: str | Path,
    output_dir: str | Path,
    physical_control_limit: int = 12,
) -> dict[str, Any]:
    """Stratify one backtest basket and persist Test A preflight artifacts."""

    backtest_path = Path(backtest_csv).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    with backtest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    stratified_rows = stratify_backtest_rows(rows)
    recommended_rows = build_recommended_inclusion_list(
        stratified_rows,
        physical_control_limit=physical_control_limit,
    )

    strata_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for row in stratified_rows:
        strata_counts[row.stratum] = strata_counts.get(row.stratum, 0) + 1
        role_counts[row.selection_role] = role_counts.get(row.selection_role, 0) + 1

    summary = {
        "backtest_csv": str(backtest_path),
        "total_markets": len(stratified_rows),
        "recommended_inclusion_count": len(recommended_rows),
        "recommended_core_non_obvious_count": sum(row.selection_role == "core_non_obvious" for row in recommended_rows),
        "recommended_control_count": sum(row.selection_role == "control" for row in recommended_rows),
        "strata_counts": strata_counts,
        "selection_role_counts": role_counts,
        "physical_control_limit": int(physical_control_limit),
        "recommended_market_ids": [row.market_id for row in recommended_rows],
        "recommended_questions": [row.question for row in recommended_rows],
        "notes": [
            "Core non-obvious strata are intentionally over-weighted before physical controls.",
            "Noise, local-weather trivia, and weak-signal climate-adjacent markets are excluded by default.",
            "Use recommended_market_ids as the inclusion list for Test A historical news runs.",
        ],
    }

    _write_json(output_path / "summary.json", summary)
    _write_json(output_path / "stratified_markets.json", [asdict(row) for row in stratified_rows])
    _write_json(output_path / "recommended_inclusion_list.json", [asdict(row) for row in recommended_rows])
    _write_csv(output_path / "stratified_markets.csv", (asdict(row) for row in stratified_rows))
    _write_csv(output_path / "recommended_inclusion_list.csv", (asdict(row) for row in recommended_rows))
    (output_path / "recommended_market_ids.txt").write_text(
        "\n".join(row.market_id for row in recommended_rows) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


__all__ = [
    "TestAMarketRow",
    "build_recommended_inclusion_list",
    "classify_test_a_market",
    "run_test_a_preflight",
    "stratify_backtest_rows",
]
