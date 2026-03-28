"""Freeman Autonomous Analyst Benchmark runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Protocol, Sequence

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freeman.agent import (  # noqa: E402
    AnalysisPipeline,
    AnalysisPipelineConfig,
    AttentionScheduler,
    AttentionTask,
    ForecastRegistry,
    ManualSignalSource,
    ProactiveEmitter,
    ShockClassification,
    Signal,
    SignalIngestionEngine,
    SignalMemory,
)
from freeman.game.runner import SimConfig  # noqa: E402
from freeman.llm import HashingEmbeddingAdapter, OllamaEmbeddingClient  # noqa: E402
from freeman.llm.deepseek import DeepSeekChatClient  # noqa: E402
from freeman.memory import KGDelta, KGEdge, KGNode, KGVectorStore, KnowledgeGraph, Reconciler, SessionLog  # noqa: E402

LOGGER = logging.getLogger("faab")

MODE_A_FULL = "MODE_A_FULL"
MODE_B_AMNESIC = "MODE_B_AMNESIC"
MODE_C_HASH = "MODE_C_HASH"
MODE_D_LLMONLY = "MODE_D_LLMONLY"
ALL_MODES = [MODE_A_FULL, MODE_B_AMNESIC, MODE_C_HASH, MODE_D_LLMONLY]

CLASSIFIER_SYSTEM_PROMPT = """FAAB_CLASSIFIER
You classify one benchmark signal for Freeman.

Return exactly one JSON object with:
- shock_type: short label
- severity: float in [0,1]
- semantic_gap: float in [0,1]
- rationale: one short sentence
"""

STATE_FORECAST_SYSTEM_PROMPT = """FAAB_STATE_FORECAST
You update a compact Freeman world state from one benchmark signal and optional retrieved memory.

All resources are on a 0-100 scale. Use only the provided resource ids and outcome ids.
Return exactly one JSON object with:
- overview: short paragraph
- key_mechanisms: list of 3-5 concise points
- resource_shocks: object mapping allowed resource ids to additive shocks in [-15, 15]
- dominant_outcome_hypothesis: string
- confidence: float in [0,1]
"""

LLM_ONLY_SYSTEM_PROMPT = """FAAB_LLM_ONLY
You are the baseline advanced LLM forecaster for the Freeman Autonomous Analyst Benchmark.

Read the domain description and available signals and return exactly one JSON object with:
- dominant_outcome: string
- confidence: float in [0,1]
- rationale: short paragraph
"""


class JSONChatClient(Protocol):
    """Protocol for benchmark LLM clients."""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark case."""

    case_id: str
    domain: str
    t0_signal: str
    t1_signal: str
    ground_truth_t2: Dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(payload["case_id"]),
            domain=str(payload["domain"]),
            t0_signal=str(payload["t0_signal"]),
            t1_signal=str(payload["t1_signal"]),
            ground_truth_t2=dict(payload["ground_truth_t2"]),
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "t0_signal": self.t0_signal,
            "t1_signal": self.t1_signal,
            "ground_truth_t2": self.ground_truth_t2,
        }


@dataclass
class StepPrediction:
    """Prediction summary for one time step."""

    step: str
    dominant_outcome: str | None
    confidence: float | None
    outcome_probs: Dict[str, float] = field(default_factory=dict)
    key_metric_value: float | None = None
    key_metric_error: float | None = None
    retrieved_node_ids: List[str] = field(default_factory=list)
    retrieved_node_count: int = 0
    retrieval_precision: float = 0.0
    trigger_mode: str = "ANALYZE"
    interest_score: float = 0.0
    autonomy_flag: bool = False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "dominant_outcome": self.dominant_outcome,
            "confidence": self.confidence,
            "outcome_probs": self.outcome_probs,
            "key_metric_value": self.key_metric_value,
            "key_metric_error": self.key_metric_error,
            "retrieved_node_ids": self.retrieved_node_ids,
            "retrieved_node_count": self.retrieved_node_count,
            "retrieval_precision": self.retrieval_precision,
            "trigger_mode": self.trigger_mode,
            "interest_score": self.interest_score,
            "autonomy_flag": self.autonomy_flag,
        }


@dataclass
class EvaluationResult:
    """Serializable result row for one case/mode pair."""

    case_id: str
    mode: str
    t0_accuracy: float | None
    t1_accuracy: float | None
    retrieval_precision: float
    autonomy_flag: bool
    status: str = "ok"
    error: str = ""
    t0_prediction: StepPrediction | None = None
    t1_prediction: StepPrediction | None = None

    def metrics_row(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "t0_accuracy": self.t0_accuracy,
            "t1_accuracy": self.t1_accuracy,
            "retrieval_precision": self.retrieval_precision,
            "autonomy_flag": self.autonomy_flag,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "t0_accuracy": self.t0_accuracy,
            "t1_accuracy": self.t1_accuracy,
            "retrieval_precision": self.retrieval_precision,
            "autonomy_flag": self.autonomy_flag,
            "status": self.status,
            "error": self.error,
            "t0_prediction": self.t0_prediction.snapshot() if self.t0_prediction is not None else None,
            "t1_prediction": self.t1_prediction.snapshot() if self.t1_prediction is not None else None,
        }


@dataclass
class RunnerConfig:
    """Benchmark configuration."""

    output_dir: Path | None = None
    retrieval_top_k: int = 8
    max_context_nodes: int = 12
    max_steps: int = 8
    seed: int = 42
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://127.0.0.1:11434"
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_timeout_seconds: float = 90.0
    shared_memory_across_cases: bool = False
    state_time_decay: float = 0.5


class HeuristicBenchmarkClient:
    """Offline-safe JSON client for smoke tests and CI."""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Dict[str, Any]:
        del temperature, max_tokens
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else "{}"
        if "FAAB_CLASSIFIER" in system:
            payload = json.loads(user)
            return self._classify(str(payload.get("text", "")))
        if "FAAB_STATE_FORECAST" in system:
            payload = json.loads(user)
            return self._state_forecast(payload)
        if "FAAB_LLM_ONLY" in system:
            payload = json.loads(user)
            return self._llm_only(payload)
        raise ValueError("Unsupported heuristic prompt.")

    def _classify(self, text: str) -> Dict[str, Any]:
        lower = text.lower()
        severity = 0.55
        semantic_gap = 0.55
        shock_type = "update"
        if any(token in lower for token in ["heatwave", "restriction", "layoff", "credit", "jealous", "cinemascore"]):
            severity = 0.85
            semantic_gap = 0.8
            shock_type = "shock"
        elif any(token in lower for token in ["therapy", "check-ins", "rain", "conservation", "holdover"]):
            severity = 0.7
            semantic_gap = 0.65
            shock_type = "adaptation"
        return {
            "shock_type": shock_type,
            "severity": severity,
            "semantic_gap": semantic_gap,
            "rationale": f"Heuristic benchmark classification for: {text[:80]}",
        }

    def _state_forecast(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        domain = str(payload.get("domain", ""))
        signal_text = str(payload.get("signal_text", "")).lower()
        retrieved = " ".join(str(item.get("content", "")) for item in payload.get("retrieved_context", []))
        merged = f"{retrieved} {signal_text}".lower()
        resource_ids = list(payload.get("resources", {}).keys())
        shocks: Dict[str, float] = {resource_id: 0.0 for resource_id in resource_ids}
        outcome = ""

        if domain == "climate_risk":
            if "rain" in merged or "conservation" in merged or "rationing eased" in merged:
                shocks.update({"policy_response": 12.0, "reservoir_storage": 6.0, "crop_stress": -5.0})
                outcome = "managed_adaptation"
            else:
                shocks.update({"precipitation_deficit": 12.0, "reservoir_storage": -10.0, "crop_stress": 9.0, "wildfire_risk": 8.0})
                outcome = "water_shortage_spiral"
        elif domain == "macroeconomy":
            if any(token in merged for token in ["layoff", "credit", "orders fell", "orders plunged", "investment froze"]):
                shocks.update({"business_demand": -12.0, "recession_risk": 11.0, "policy_rate": 5.0, "inflation_pressure": 3.0})
                outcome = "recession_spiral"
            else:
                shocks.update({"trade_cost_pressure": 11.0, "inflation_pressure": 10.0, "policy_rate": 6.0})
                outcome = "inflation_persistence"
        elif domain == "social_relationships":
            if any(token in merged for token in ["therapy", "honest conversation", "check-ins", "apologized", "counseling"]):
                shocks.update({"communication_quality": 11.0, "trust_level": 8.0, "repair_capacity": 10.0, "breakup_risk": -8.0, "jealousy_pressure": -4.0})
                outcome = "repair_path"
            elif any(token in merged for token in ["stonewall", "ghost", "affair", "moved out"]):
                shocks.update({"trust_level": -12.0, "communication_quality": -10.0, "breakup_risk": 12.0, "jealousy_pressure": 7.0})
                outcome = "breakup_path"
            else:
                shocks.update({"work_stress": 8.0, "jealousy_pressure": 8.0, "trust_level": -6.0, "communication_quality": -5.0})
                outcome = "fragile_stalemate"
        elif domain == "film_release":
            if any(token in merged for token in ["weekday drop", "front-loaded", "cinemascore", "mixed exits"]):
                shocks.update({"opening_weekend": 12.0, "box_office_legs": -11.0, "critic_sentiment": -5.0, "word_of_mouth": -6.0})
                outcome = "front_loaded_opening"
            elif any(token in merged for token in ["holdover", "strong exits", "word of mouth", "repeat viewing"]):
                shocks.update({"opening_weekend": 8.0, "box_office_legs": 10.0, "word_of_mouth": 9.0, "buzz": 6.0})
                outcome = "breakout_hit"
            else:
                shocks.update({"buzz": 9.0, "marketing_intensity": 8.0, "opening_weekend": 7.0})
                outcome = "breakout_hit"
        confidence = 0.72 if outcome else 0.45
        return {
            "overview": f"Heuristic state update for {domain}.",
            "key_mechanisms": [f"{resource_id} adjusted by {shock:.1f}" for resource_id, shock in shocks.items() if abs(shock) > 0.0][:5],
            "resource_shocks": shocks,
            "dominant_outcome_hypothesis": outcome,
            "confidence": confidence,
        }

    def _llm_only(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        domain = str(payload.get("domain", ""))
        signal_text = str(payload.get("signal_text", "")).lower()
        prior_text = str(payload.get("prior_signal", "")).lower()
        merged = f"{prior_text} {signal_text}"
        response = self._state_forecast(
            {
                "domain": domain,
                "signal_text": merged,
                "retrieved_context": [],
                "resources": _resource_catalog(_domain_template(domain, case_id="heuristic")),
            }
        )
        return {
            "dominant_outcome": response.get("dominant_outcome_hypothesis"),
            "confidence": response.get("confidence", 0.5),
            "rationale": response.get("overview", ""),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_api_key(path: str | Path | None = None) -> str:
    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    key_path = Path(path or REPO_ROOT / "DS.txt").resolve()
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"DeepSeek API key file is empty: {key_path}")
    return key


def load_cases(path: str | Path) -> List[BenchmarkCase]:
    """Load benchmark cases from JSONL."""

    cases: List[BenchmarkCase] = []
    source = Path(path).resolve()
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cases.append(BenchmarkCase.from_payload(json.loads(stripped)))
    return cases


def _base_run_dir(output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        target = Path(output_root).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target
    target = (REPO_ROOT / "runs" / f"faab_eval_{_timestamp_slug()}").resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _resource_catalog(schema: Dict[str, Any]) -> Dict[str, str]:
    return {resource["id"]: resource["name"] for resource in schema["resources"]}


def _domain_description(domain: str) -> str:
    descriptions = {
        "climate_risk": "Hydro-climate stress, reservoir management, crop losses, and adaptation under drought and heat.",
        "macroeconomy": "Trade-cost shocks, inflation pressure, monetary tightening, and recession risk.",
        "social_relationships": "Trust, communication, jealousy, emotional exhaustion, and repair versus breakup risk.",
        "film_release": "Buzz, critics, opening weekend strength, word of mouth, and box-office legs.",
    }
    if domain not in descriptions:
        raise KeyError(f"Unsupported domain: {domain}")
    return descriptions[domain]


def _domain_template(domain: str, *, case_id: str) -> Dict[str, Any]:
    if domain == "climate_risk":
        return {
            "domain_id": case_id,
            "name": "Climate Water Stress Template",
            "description": "Hydro-climate template for drought, storage, crop stress, and adaptation.",
            "metadata": {"benchmark_domain": domain},
            "actors": [
                {"id": "utilities", "name": "Water Utilities", "state": {"preparedness": 0.55}},
                {"id": "farmers", "name": "Farmers", "state": {"resilience": 0.5}},
                {"id": "state", "name": "State Agency", "state": {"coordination": 0.48}},
            ],
            "resources": [
                {
                    "id": "precipitation_deficit",
                    "name": "Precipitation Deficit",
                    "value": 34.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.88, "c": 2.0, "coupling_weights": {"policy_response": -0.02}},
                },
                {
                    "id": "reservoir_storage",
                    "name": "Reservoir Storage",
                    "value": 64.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.90,
                        "c": 0.0,
                        "coupling_weights": {"precipitation_deficit": -0.08, "policy_response": 0.06},
                    },
                },
                {
                    "id": "crop_stress",
                    "name": "Crop Stress",
                    "value": 28.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.84,
                        "c": 1.5,
                        "coupling_weights": {"precipitation_deficit": 0.07, "reservoir_storage": -0.05},
                    },
                },
                {
                    "id": "policy_response",
                    "name": "Policy Response",
                    "value": 24.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.86,
                        "c": 1.0,
                        "coupling_weights": {"precipitation_deficit": 0.06, "crop_stress": 0.04},
                    },
                },
                {
                    "id": "wildfire_risk",
                    "name": "Wildfire Risk",
                    "value": 21.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.82,
                        "c": 1.0,
                        "coupling_weights": {"precipitation_deficit": 0.08, "reservoir_storage": -0.03},
                    },
                },
            ],
            "relations": [
                {"source_id": "state", "target_id": "utilities", "relation_type": "regulates", "weights": {"strength": 0.7}},
                {"source_id": "utilities", "target_id": "farmers", "relation_type": "allocates", "weights": {"strength": 0.8}},
            ],
            "outcomes": [
                {
                    "id": "managed_adaptation",
                    "label": "Managed Adaptation",
                    "scoring_weights": {
                        "policy_response": 0.09,
                        "reservoir_storage": 0.08,
                        "crop_stress": -0.07,
                        "wildfire_risk": -0.05,
                    },
                },
                {
                    "id": "water_shortage_spiral",
                    "label": "Water Shortage Spiral",
                    "scoring_weights": {
                        "precipitation_deficit": 0.08,
                        "reservoir_storage": -0.09,
                        "crop_stress": 0.08,
                        "wildfire_risk": 0.06,
                    },
                },
                {
                    "id": "agricultural_loss_cycle",
                    "label": "Agricultural Loss Cycle",
                    "scoring_weights": {
                        "crop_stress": 0.1,
                        "reservoir_storage": -0.05,
                        "policy_response": -0.03,
                    },
                },
            ],
            "causal_dag": [
                {"source": "precipitation_deficit", "target": "reservoir_storage", "expected_sign": "-", "strength": "strong"},
                {"source": "precipitation_deficit", "target": "crop_stress", "expected_sign": "+", "strength": "strong"},
                {"source": "reservoir_storage", "target": "crop_stress", "expected_sign": "-", "strength": "strong"},
                {"source": "policy_response", "target": "reservoir_storage", "expected_sign": "+", "strength": "weak"},
                {"source": "precipitation_deficit", "target": "wildfire_risk", "expected_sign": "+", "strength": "strong"},
            ],
        }
    if domain == "macroeconomy":
        return {
            "domain_id": case_id,
            "name": "Trade Shock Macro Template",
            "description": "Macro template for imported inflation and recession risk.",
            "metadata": {"benchmark_domain": domain},
            "actors": [
                {"id": "households", "name": "Households", "state": {"confidence": 0.55}},
                {"id": "firms", "name": "Firms", "state": {"pricing_power": 0.55}},
                {"id": "central_bank", "name": "Central Bank", "state": {"hawkishness": 0.6}},
            ],
            "resources": [
                {
                    "id": "trade_cost_pressure",
                    "name": "Trade Cost Pressure",
                    "value": 40.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.84, "c": 2.5, "coupling_weights": {"business_demand": -0.03}},
                },
                {
                    "id": "inflation_pressure",
                    "name": "Inflation Pressure",
                    "value": 35.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.83,
                        "c": 1.5,
                        "coupling_weights": {"trade_cost_pressure": 0.11, "policy_rate": -0.09, "business_demand": 0.02},
                    },
                },
                {
                    "id": "business_demand",
                    "name": "Business Demand",
                    "value": 58.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.88, "c": 2.0, "coupling_weights": {"inflation_pressure": -0.05, "policy_rate": -0.06}},
                },
                {
                    "id": "policy_rate",
                    "name": "Policy Rate Stance",
                    "value": 28.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.80, "c": 1.0, "coupling_weights": {"inflation_pressure": 0.08, "business_demand": -0.03}},
                },
                {
                    "id": "recession_risk",
                    "name": "Recession Risk",
                    "value": 24.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.84,
                        "c": 1.0,
                        "coupling_weights": {"business_demand": -0.07, "policy_rate": 0.04, "trade_cost_pressure": 0.05},
                    },
                },
            ],
            "relations": [
                {"source_id": "households", "target_id": "firms", "relation_type": "demand_link", "weights": {"strength": 0.5}},
                {"source_id": "central_bank", "target_id": "firms", "relation_type": "policy_link", "weights": {"strength": 0.6}},
            ],
            "outcomes": [
                {
                    "id": "soft_landing",
                    "label": "Soft Landing",
                    "scoring_weights": {"business_demand": 0.08, "inflation_pressure": -0.06, "recession_risk": -0.08},
                },
                {
                    "id": "inflation_persistence",
                    "label": "Inflation Persistence",
                    "scoring_weights": {"inflation_pressure": 0.09, "trade_cost_pressure": 0.06, "policy_rate": 0.03, "business_demand": -0.03},
                },
                {
                    "id": "recession_spiral",
                    "label": "Recession Spiral",
                    "scoring_weights": {"recession_risk": 0.1, "business_demand": -0.08, "policy_rate": 0.03},
                    "regime_shifts": [
                        {
                            "condition": "business_demand <= -5 AND policyrate >= 5",
                            "multiplier": 3.0,
                        }
                    ],
                },
            ],
            "causal_dag": [
                {"source": "trade_cost_pressure", "target": "inflation_pressure", "expected_sign": "+", "strength": "strong"},
                {"source": "inflation_pressure", "target": "business_demand", "expected_sign": "-", "strength": "strong"},
                {"source": "policy_rate", "target": "business_demand", "expected_sign": "-", "strength": "strong"},
                {"source": "inflation_pressure", "target": "policy_rate", "expected_sign": "+", "strength": "strong"},
                {"source": "business_demand", "target": "recession_risk", "expected_sign": "-", "strength": "strong"},
                {"source": "trade_cost_pressure", "target": "recession_risk", "expected_sign": "+", "strength": "weak"},
            ],
        }
    if domain == "social_relationships":
        return {
            "domain_id": case_id,
            "name": "Relationship Stress Template",
            "description": "Relationship template for trust, repair, and breakup risk.",
            "metadata": {"benchmark_domain": domain},
            "actors": [
                {"id": "partner_a", "name": "Partner A", "state": {"patience": 0.5}},
                {"id": "partner_b", "name": "Partner B", "state": {"patience": 0.5}},
            ],
            "resources": [
                {
                    "id": "work_stress",
                    "name": "Work Stress",
                    "value": 42.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.84, "c": 2.0, "coupling_weights": {"repair_capacity": -0.03}},
                },
                {
                    "id": "communication_quality",
                    "name": "Communication Quality",
                    "value": 58.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.82,
                        "c": 2.0,
                        "coupling_weights": {"work_stress": -0.07, "trust_level": 0.05, "jealousy_pressure": -0.05},
                    },
                },
                {
                    "id": "trust_level",
                    "name": "Trust Level",
                    "value": 61.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.84,
                        "c": 1.5,
                        "coupling_weights": {"communication_quality": 0.06, "jealousy_pressure": -0.08},
                    },
                },
                {
                    "id": "jealousy_pressure",
                    "name": "Jealousy Pressure",
                    "value": 28.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.8,
                        "c": 1.0,
                        "coupling_weights": {"communication_quality": -0.04, "trust_level": -0.05},
                    },
                },
                {
                    "id": "repair_capacity",
                    "name": "Repair Capacity",
                    "value": 52.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.82,
                        "c": 2.0,
                        "coupling_weights": {"communication_quality": 0.05, "work_stress": -0.04},
                    },
                },
                {
                    "id": "breakup_risk",
                    "name": "Breakup Risk",
                    "value": 22.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.84,
                        "c": 1.0,
                        "coupling_weights": {"trust_level": -0.08, "communication_quality": -0.06, "jealousy_pressure": 0.07, "work_stress": 0.04},
                    },
                },
            ],
            "relations": [
                {"source_id": "partner_a", "target_id": "partner_b", "relation_type": "romantic_link", "weights": {"strength": 0.9}},
                {"source_id": "partner_b", "target_id": "partner_a", "relation_type": "romantic_link", "weights": {"strength": 0.9}},
            ],
            "outcomes": [
                {
                    "id": "repair_path",
                    "label": "Repair Path",
                    "scoring_weights": {"repair_capacity": 0.08, "trust_level": 0.07, "communication_quality": 0.06, "breakup_risk": -0.1},
                },
                {
                    "id": "fragile_stalemate",
                    "label": "Fragile Stalemate",
                    "scoring_weights": {"work_stress": 0.05, "jealousy_pressure": 0.04, "communication_quality": -0.02, "trust_level": -0.02},
                },
                {
                    "id": "breakup_path",
                    "label": "Breakup Path",
                    "scoring_weights": {"breakup_risk": 0.1, "jealousy_pressure": 0.05, "trust_level": -0.07, "communication_quality": -0.06},
                },
            ],
            "causal_dag": [
                {"source": "work_stress", "target": "communication_quality", "expected_sign": "-", "strength": "strong"},
                {"source": "communication_quality", "target": "trust_level", "expected_sign": "+", "strength": "strong"},
                {"source": "trust_level", "target": "jealousy_pressure", "expected_sign": "-", "strength": "strong"},
                {"source": "jealousy_pressure", "target": "breakup_risk", "expected_sign": "+", "strength": "strong"},
                {"source": "repair_capacity", "target": "work_stress", "expected_sign": "-", "strength": "weak"},
                {"source": "communication_quality", "target": "repair_capacity", "expected_sign": "+", "strength": "strong"},
            ],
        }
    if domain == "film_release":
        return {
            "domain_id": case_id,
            "name": "Film Release Template",
            "description": "Film release template for buzz, reviews, and box-office dynamics.",
            "metadata": {"benchmark_domain": domain},
            "actors": [
                {"id": "studio", "name": "Studio", "state": {"confidence": 0.55}},
                {"id": "audience", "name": "Audience", "state": {"engagement": 0.6}},
            ],
            "resources": [
                {
                    "id": "buzz",
                    "name": "Audience Buzz",
                    "value": 48.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.84, "c": 2.0, "coupling_weights": {"critic_sentiment": 0.04, "marketing_intensity": 0.05}},
                },
                {
                    "id": "critic_sentiment",
                    "name": "Critic Sentiment",
                    "value": 50.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.88, "c": 0.0, "coupling_weights": {}},
                },
                {
                    "id": "marketing_intensity",
                    "name": "Marketing Intensity",
                    "value": 44.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.84, "c": 2.0, "coupling_weights": {"buzz": 0.02}},
                },
                {
                    "id": "word_of_mouth",
                    "name": "Word of Mouth",
                    "value": 39.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {"a": 0.82, "c": 1.5, "coupling_weights": {"critic_sentiment": 0.07, "buzz": 0.05}},
                },
                {
                    "id": "opening_weekend",
                    "name": "Opening Weekend Strength",
                    "value": 34.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.8,
                        "c": 2.0,
                        "coupling_weights": {"buzz": 0.08, "marketing_intensity": 0.06, "critic_sentiment": 0.03},
                    },
                },
                {
                    "id": "box_office_legs",
                    "name": "Box Office Legs",
                    "value": 30.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.82,
                        "c": 1.0,
                        "coupling_weights": {"word_of_mouth": 0.08, "critic_sentiment": 0.05, "opening_weekend": 0.03},
                    },
                },
            ],
            "relations": [
                {"source_id": "studio", "target_id": "audience", "relation_type": "promotion_link", "weights": {"strength": 0.7}},
            ],
            "outcomes": [
                {
                    "id": "breakout_hit",
                    "label": "Breakout Hit",
                    "scoring_weights": {"opening_weekend": 0.06, "box_office_legs": 0.08, "word_of_mouth": 0.06, "critic_sentiment": 0.04},
                },
                {
                    "id": "front_loaded_opening",
                    "label": "Front-Loaded Opening",
                    "scoring_weights": {"opening_weekend": 0.08, "box_office_legs": -0.07, "buzz": 0.05, "critic_sentiment": -0.04},
                    "regime_shifts": [
                        {
                            "condition": "criticsentiment <= -5 AND boxofficelegs <= -5",
                            "multiplier": 3.0,
                        }
                    ],
                },
                {
                    "id": "underperformer",
                    "label": "Underperformer",
                    "scoring_weights": {"buzz": -0.04, "critic_sentiment": -0.08, "opening_weekend": -0.06, "box_office_legs": -0.08},
                },
            ],
            "causal_dag": [
                {"source": "marketing_intensity", "target": "buzz", "expected_sign": "+", "strength": "strong"},
                {"source": "critic_sentiment", "target": "word_of_mouth", "expected_sign": "+", "strength": "strong"},
                {"source": "buzz", "target": "opening_weekend", "expected_sign": "+", "strength": "strong"},
                {"source": "word_of_mouth", "target": "box_office_legs", "expected_sign": "+", "strength": "strong"},
                {"source": "critic_sentiment", "target": "box_office_legs", "expected_sign": "+", "strength": "strong"},
            ],
        }
    raise KeyError(f"Unsupported domain: {domain}")


def _clamp_resource_shocks(schema: Dict[str, Any], proposed: Dict[str, Any]) -> Dict[str, float]:
    allowed = set(_resource_catalog(schema))
    shocks: Dict[str, float] = {}
    for resource_id, value in proposed.items():
        if resource_id not in allowed:
            continue
        shocks[resource_id] = float(min(max(float(value), -15.0), 15.0))
    return shocks


def _apply_resource_shocks(schema: Dict[str, Any], resource_shocks: Dict[str, float]) -> Dict[str, Any]:
    updated = json.loads(json.dumps(schema))
    for resource in updated["resources"]:
        shock = float(resource_shocks.get(resource["id"], 0.0))
        lower = float(resource.get("min_value", 0.0))
        upper = float(resource.get("max_value", 100.0))
        resource["value"] = float(min(max(float(resource["value"]) + shock, lower), upper))
    return updated


def _extract_key_metric(simulation: Dict[str, Any], metric_name: str | None) -> float | None:
    if not metric_name:
        return None
    trajectory = simulation.get("trajectory", [])
    if not trajectory:
        return None
    resources = trajectory[-1].get("resources", {})
    if metric_name in resources and "value" in resources[metric_name]:
        return float(resources[metric_name]["value"])
    return None


def _accuracy(predicted: str | None, ground_truth: Dict[str, Any]) -> float | None:
    if predicted is None:
        return None
    target = str(ground_truth.get("dominant_outcome", "")).strip()
    if not target:
        return None
    return 1.0 if predicted == target else 0.0


def _default_trace_payload(case: BenchmarkCase, mode: str) -> Dict[str, Any]:
    return {
        "case": case.snapshot(),
        "mode": mode,
        "created_at": _utc_now(),
        "llm_calls": [],
        "steps": {},
        "errors": [],
    }


class BenchmarkRunner:
    """Evaluate Freeman against several longitudinal baselines."""

    def __init__(
        self,
        *,
        mode: str,
        output_dir: str | Path | None = None,
        llm_client: JSONChatClient | None = None,
        config: RunnerConfig | None = None,
        semantic_embedding_adapter: Any | None = None,
    ) -> None:
        if mode not in ALL_MODES:
            raise ValueError(f"Unsupported benchmark mode: {mode}")
        base_output = _base_run_dir(output_dir)
        self.config = config or RunnerConfig(output_dir=base_output)
        self.config.output_dir = base_output
        self.mode = mode
        self.run_dir = self.config.output_dir
        self.traces_dir = self.run_dir / "traces"
        self.snapshots_dir = self.run_dir / "kg_snapshots"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client or self._build_live_client()
        self.semantic_embedding_adapter = semantic_embedding_adapter
        self._state: Dict[str, Any] = {}
        self._active_case_id: str | None = None
        self._state_generation = 0

    def _build_live_client(self) -> JSONChatClient:
        return DeepSeekChatClient(
            api_key=_load_api_key(),
            model=self.config.deepseek_model,
            base_url=self.config.deepseek_base_url,
            timeout_seconds=self.config.deepseek_timeout_seconds,
            max_retries=3,
            retry_backoff_seconds=2.0,
        )

    def _build_embedding_adapter(self) -> Any | None:
        if self.mode == MODE_D_LLMONLY:
            return None
        if self.mode == MODE_C_HASH:
            return HashingEmbeddingAdapter()
        if self.semantic_embedding_adapter is not None:
            return self.semantic_embedding_adapter
        return OllamaEmbeddingClient(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
        )

    def _initialize_case_state(self, case: BenchmarkCase) -> None:
        self._state_generation += 1
        case_root = self.run_dir / "_state" / self.mode / f"{_slugify(case.case_id)}__{self._state_generation:03d}"
        case_root.mkdir(parents=True, exist_ok=True)
        embedding_adapter = self._build_embedding_adapter()
        vectorstore = None
        knowledge_graph = None
        forecast_registry = None
        pipeline = None
        if self.mode != MODE_D_LLMONLY:
            vectorstore = KGVectorStore(path=case_root / "chroma_db", collection_name=f"{_slugify(self.mode)}_{_slugify(case.case_id)}")
            knowledge_graph = KnowledgeGraph(
                json_path=case_root / "kg.json",
                auto_load=False,
                auto_save=True,
                llm_adapter=embedding_adapter,
                vectorstore=vectorstore,
            )
            forecast_registry = ForecastRegistry(
                json_path=case_root / "forecasts.json",
                auto_load=False,
                auto_save=True,
            )
            pipeline = AnalysisPipeline(
                sim_config=SimConfig(
                    max_steps=self.config.max_steps,
                    level2_check_every=1,
                    stop_on_hard_level2=False,
                    convergence_check_steps=100,
                    convergence_epsilon=3.0e-2,
                    seed=self.config.seed,
                ),
                knowledge_graph=knowledge_graph,
                reconciler=Reconciler(),
                forecast_registry=forecast_registry,
                emitter=ProactiveEmitter(),
                config=AnalysisPipelineConfig(
                    retrieval_top_k=self.config.retrieval_top_k,
                    max_context_nodes=self.config.max_context_nodes,
                ),
            )
        self._state = {
            "case_root": case_root,
            "embedding_adapter": embedding_adapter,
            "vectorstore": vectorstore,
            "knowledge_graph": knowledge_graph,
            "forecast_registry": forecast_registry,
            "pipeline": pipeline,
            "base_world": None,
            "current_world": None,
            "ingestion": SignalIngestionEngine(),
            "signal_memory": SignalMemory(),
            "scheduler": AttentionScheduler(attention_budget=2.0, ucb_beta=0.0),
        }
        self._active_case_id = case.case_id

    def _ensure_case_state(self, case: BenchmarkCase) -> None:
        if self._active_case_id == case.case_id and self._state:
            return
        if self.config.shared_memory_across_cases and self._active_case_id == case.case_id and self._state:
            return
        if self.config.shared_memory_across_cases and self._state and self._active_case_id is not None:
            return
        self._initialize_case_state(case)

    def _clear_memory(self, case: BenchmarkCase) -> None:
        LOGGER.info("case=%s mode=%s clearing memory before T1", case.case_id, self.mode)
        self._initialize_case_state(case)

    def _trace_chat_json(
        self,
        *,
        trace: Dict[str, Any],
        label: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        record = {
            "label": label,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "started_at": _utc_now(),
        }
        try:
            response = self.llm_client.chat_json(messages, temperature=temperature, max_tokens=max_tokens)
            record["response"] = response
            record["ended_at"] = _utc_now()
            trace["llm_calls"].append(record)
            return response
        except Exception as exc:  # noqa: BLE001
            record["error"] = str(exc)
            record["ended_at"] = _utc_now()
            trace["llm_calls"].append(record)
            raise

    def _classify_signal(self, signal: Signal, trace: Dict[str, Any], *, step: str) -> ShockClassification:
        payload = {
            "signal_id": signal.signal_id,
            "topic": signal.topic,
            "text": signal.text,
            "entities": signal.entities,
            "sentiment": signal.sentiment,
        }
        response = self._trace_chat_json(
            trace=trace,
            label=f"{step}_classifier",
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return ShockClassification(
            shock_type=str(response.get("shock_type", "update")),
            severity=float(min(max(float(response.get("severity", 0.5)), 0.0), 1.0)),
            semantic_gap=float(min(max(float(response.get("semantic_gap", 0.5)), 0.0), 1.0)),
            rationale=str(response.get("rationale", "")).strip(),
        )

    def _make_signal(self, case: BenchmarkCase, *, step: str, text: str) -> Signal:
        return Signal(
            signal_id=f"{case.case_id}:{step}",
            source_type="benchmark",
            text=text,
            topic=case.domain,
            entities=[case.domain.replace("_", " ")],
            sentiment=0.0,
            metadata={"case_id": case.case_id, "domain": case.domain, "step": step},
        )

    def _schedule_signal(
        self,
        *,
        signal: Signal,
        classification: ShockClassification,
        case: BenchmarkCase,
        step: str,
    ) -> tuple[str, float]:
        ingestion = self._state["ingestion"]
        signal_memory = self._state["signal_memory"]
        scheduler = self._state["scheduler"]
        triggers = ingestion.ingest(
            ManualSignalSource([signal]),
            classifier=lambda _: classification,
            signal_memory=signal_memory,
        )
        trigger_mode = triggers[0].mode if triggers else "ANALYZE"
        trigger_score = triggers[0].mahalanobis_score if triggers else 0.0
        task_id = f"{case.case_id}:{step}"
        scheduler.add_task(
            AttentionTask(
                task_id=task_id,
                description=signal.text,
                expected_information_gain=max(classification.severity, 0.1),
                cost=1.0,
                anomaly_score=trigger_score,
                semantic_gap=classification.semantic_gap,
                metadata={"case_id": case.case_id, "domain": case.domain, "step": step},
            )
        )
        decision = scheduler.select_task()
        interest = decision.interest_score if decision is not None else 0.0
        return trigger_mode, float(interest)

    def _retrieve_context(self, signal_text: str) -> List[KGNode]:
        knowledge_graph: KnowledgeGraph | None = self._state.get("knowledge_graph")
        if knowledge_graph is None:
            return []
        nodes = knowledge_graph.semantic_query(signal_text, top_k=self.config.retrieval_top_k)
        return nodes[: self.config.max_context_nodes]

    def _retrieval_precision(self, case: BenchmarkCase, nodes: Sequence[KGNode]) -> float:
        if not nodes:
            return 0.0
        relevant = 0
        for node in nodes:
            if (
                str(node.metadata.get("case_id", "")) == case.case_id
                or str(node.metadata.get("domain_id", "")) == case.case_id
                or case.case_id in node.id
            ):
                relevant += 1
        return float(relevant / len(nodes))

    def _infer_state(
        self,
        *,
        case: BenchmarkCase,
        step: str,
        signal: Signal,
        classification: ShockClassification,
        retrieved_nodes: Sequence[KGNode],
        trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        schema = _domain_template(case.domain, case_id=case.case_id)
        payload = {
            "case_id": case.case_id,
            "domain": case.domain,
            "domain_description": _domain_description(case.domain),
            "step": step,
            "signal_text": signal.text,
            "classification": asdict(classification),
            "resources": _resource_catalog(schema),
            "outcomes": [outcome["id"] for outcome in schema["outcomes"]],
            "retrieved_context": [
                {
                    "id": node.id,
                    "label": node.label,
                    "content": node.content,
                    "metadata": node.metadata,
                }
                for node in retrieved_nodes
            ],
        }
        response = self._trace_chat_json(
            trace=trace,
            label=f"{step}_state_inference",
            messages=[
                {"role": "system", "content": STATE_FORECAST_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        return {
            "overview": str(response.get("overview", "")).strip(),
            "key_mechanisms": [str(item).strip() for item in response.get("key_mechanisms", []) if str(item).strip()],
            "resource_shocks": _clamp_resource_shocks(schema, dict(response.get("resource_shocks", {}))),
            "dominant_outcome_hypothesis": str(response.get("dominant_outcome_hypothesis", "")).strip(),
            "confidence": float(min(max(float(response.get("confidence", 0.0)), 0.0), 1.0)),
        }

    def _llm_only_prediction(
        self,
        *,
        case: BenchmarkCase,
        step: str,
        signal_text: str,
        prior_signal: str,
        trace: Dict[str, Any],
    ) -> StepPrediction:
        schema = _domain_template(case.domain, case_id=case.case_id)
        payload = {
            "case_id": case.case_id,
            "domain": case.domain,
            "domain_description": _domain_description(case.domain),
            "step": step,
            "prior_signal": prior_signal,
            "signal_text": signal_text,
            "allowed_outcomes": [outcome["id"] for outcome in schema["outcomes"]],
        }
        response = self._trace_chat_json(
            trace=trace,
            label=f"{step}_llm_only_forecast",
            messages=[
                {"role": "system", "content": LLM_ONLY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        metric_name = str(case.ground_truth_t2.get("key_metric_name", ""))
        return StepPrediction(
            step=step,
            dominant_outcome=str(response.get("dominant_outcome", "")).strip() or None,
            confidence=float(min(max(float(response.get("confidence", 0.0)), 0.0), 1.0)),
            outcome_probs={},
            key_metric_value=None if not metric_name else None,
            key_metric_error=None,
            retrieved_node_ids=[],
            retrieved_node_count=0,
            retrieval_precision=0.0,
            trigger_mode="PROMPT_ONLY",
            interest_score=0.0,
            autonomy_flag=False,
        )

    def _persist_signal_and_inference(
        self,
        *,
        case: BenchmarkCase,
        step: str,
        signal: Signal,
        classification: ShockClassification,
        inference: Dict[str, Any],
        session_log: SessionLog,
    ) -> None:
        knowledge_graph: KnowledgeGraph = self._state["knowledge_graph"]
        signal_node = KGNode(
            id=f"{case.case_id}:{step}:signal",
            label=f"{case.domain} signal {step.upper()}",
            node_type="signal",
            content=signal.text,
            confidence=max(classification.severity, 0.25),
            metadata={
                "case_id": case.case_id,
                "domain": case.domain,
                "step": step,
                "classification": asdict(classification),
            },
        )
        inference_node = KGNode(
            id=f"{case.case_id}:{step}:inference",
            label=f"{case.domain} inference {step.upper()}",
            node_type="state_inference",
            content=inference["overview"],
            confidence=max(float(inference["confidence"]), 0.25),
            metadata={
                "case_id": case.case_id,
                "domain": case.domain,
                "step": step,
                "dominant_outcome_hypothesis": inference["dominant_outcome_hypothesis"],
                "resource_shocks": inference["resource_shocks"],
                "key_mechanisms": inference["key_mechanisms"],
            },
        )
        knowledge_graph.add_node(signal_node)
        knowledge_graph.add_node(inference_node)
        knowledge_graph.add_edge(
            KGEdge(
                source=signal_node.id,
                target=inference_node.id,
                relation_type="supports",
                confidence=max(classification.severity, 0.25),
                weight=1.0,
                metadata={"case_id": case.case_id, "step": step},
            )
        )
        session_log.add_kg_delta(
            KGDelta(
                operation="add_node",
                target_id=signal_node.id,
                payload={"node": signal_node.snapshot()},
                support=max(1, int(round(classification.severity * 5))),
            )
        )
        session_log.add_kg_delta(
            KGDelta(
                operation="add_node",
                target_id=inference_node.id,
                payload={"node": inference_node.snapshot()},
                support=max(1, int(round(float(inference["confidence"]) * 5))),
            )
        )

    def _run_freeman_step(
        self,
        *,
        case: BenchmarkCase,
        step: str,
        signal_text: str,
        trace: Dict[str, Any],
    ) -> StepPrediction:
        signal = self._make_signal(case, step=step, text=signal_text)
        classification = self._classify_signal(signal, trace, step=step)
        trigger_mode, interest_score = self._schedule_signal(signal=signal, classification=classification, case=case, step=step)
        retrieved_nodes = self._retrieve_context(signal.text) if step == "t1" else []
        retrieval_precision = self._retrieval_precision(case, retrieved_nodes)
        inference = self._infer_state(
            case=case,
            step=step,
            signal=signal,
            classification=classification,
            retrieved_nodes=retrieved_nodes,
            trace=trace,
        )
        session_log = SessionLog(session_id=f"faab:{case.case_id}:{step}")
        self._persist_signal_and_inference(
            case=case,
            step=step,
            signal=signal,
            classification=classification,
            inference=inference,
            session_log=session_log,
        )
        pipeline: AnalysisPipeline = self._state["pipeline"]
        base_world = self._state.get("base_world")
        if base_world is None:
            base_world = pipeline.compiler.compile(_domain_template(case.domain, case_id=case.case_id))
            self._state["base_world"] = base_world.clone()
        prior_world = self._state.get("current_world") or base_world.clone()
        time_decay = self.config.state_time_decay if step != "t0" else 1.0
        shocked_world = prior_world.apply_shocks(
            inference["resource_shocks"],
            time_decay=time_decay,
        )
        pipeline: AnalysisPipeline = self._state["pipeline"]
        result = pipeline.run(shocked_world, session_log=session_log)
        self._state["current_world"] = shocked_world.clone()
        metric_name = str(case.ground_truth_t2.get("key_metric_name", "")).strip()
        key_metric_value = _extract_key_metric(result.simulation, metric_name)
        metric_target = case.ground_truth_t2.get("key_metric")
        key_metric_error = None
        if key_metric_value is not None and isinstance(metric_target, (int, float)):
            key_metric_error = float(abs(key_metric_value - float(metric_target)))
        return StepPrediction(
            step=step,
            dominant_outcome=result.dominant_outcome,
            confidence=float(result.simulation.get("confidence", 0.0)),
            outcome_probs={key: float(value) for key, value in result.simulation.get("final_outcome_probs", {}).items()},
            key_metric_value=key_metric_value,
            key_metric_error=key_metric_error,
            retrieved_node_ids=[node.id for node in retrieved_nodes],
            retrieved_node_count=len(retrieved_nodes),
            retrieval_precision=retrieval_precision,
            trigger_mode=trigger_mode,
            interest_score=interest_score,
            autonomy_flag=bool(step == "t1" and len(retrieved_nodes) > 0),
        )

    def _write_trace(self, case: BenchmarkCase, trace: Dict[str, Any]) -> Path:
        target = self.traces_dir / f"{_slugify(case.case_id)}__{_slugify(self.mode)}.json"
        target.write_text(json.dumps(trace, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return target

    def _write_kg_snapshot(self, case: BenchmarkCase) -> Path:
        target = self.snapshots_dir / f"{_slugify(case.case_id)}__{_slugify(self.mode)}__t1.json"
        knowledge_graph: KnowledgeGraph | None = self._state.get("knowledge_graph")
        if knowledge_graph is None:
            payload = {"backend": "none", "mode": self.mode, "case_id": case.case_id}
            target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return target
        knowledge_graph.save(target)
        return target

    def evaluate_case(self, case: BenchmarkCase | Dict[str, Any]) -> EvaluationResult:
        """Evaluate one case under the runner mode."""

        benchmark_case = case if isinstance(case, BenchmarkCase) else BenchmarkCase.from_payload(case)
        trace = _default_trace_payload(benchmark_case, self.mode)
        try:
            self._ensure_case_state(benchmark_case)
            if self.mode == MODE_D_LLMONLY:
                t0_prediction = self._llm_only_prediction(
                    case=benchmark_case,
                    step="t0",
                    signal_text=benchmark_case.t0_signal,
                    prior_signal="",
                    trace=trace,
                )
                t1_prediction = self._llm_only_prediction(
                    case=benchmark_case,
                    step="t1",
                    signal_text=benchmark_case.t1_signal,
                    prior_signal=benchmark_case.t0_signal,
                    trace=trace,
                )
            else:
                t0_prediction = self._run_freeman_step(
                    case=benchmark_case,
                    step="t0",
                    signal_text=benchmark_case.t0_signal,
                    trace=trace,
                )
                if self.mode == MODE_B_AMNESIC:
                    self._clear_memory(benchmark_case)
                t1_prediction = self._run_freeman_step(
                    case=benchmark_case,
                    step="t1",
                    signal_text=benchmark_case.t1_signal,
                    trace=trace,
                )
            trace["steps"]["t0"] = t0_prediction.snapshot()
            trace["steps"]["t1"] = t1_prediction.snapshot()
            snapshot_path = self._write_kg_snapshot(benchmark_case)
            trace["kg_snapshot_path"] = str(snapshot_path)
            self._write_trace(benchmark_case, trace)
            return EvaluationResult(
                case_id=benchmark_case.case_id,
                mode=self.mode,
                t0_accuracy=_accuracy(t0_prediction.dominant_outcome, benchmark_case.ground_truth_t2),
                t1_accuracy=_accuracy(t1_prediction.dominant_outcome, benchmark_case.ground_truth_t2),
                retrieval_precision=float(t1_prediction.retrieval_precision),
                autonomy_flag=bool(t1_prediction.autonomy_flag),
                t0_prediction=t0_prediction,
                t1_prediction=t1_prediction,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("case=%s mode=%s failed: %s", benchmark_case.case_id, self.mode, exc)
            trace["errors"].append({"timestamp": _utc_now(), "message": str(exc)})
            self._write_trace(benchmark_case, trace)
            try:
                self._write_kg_snapshot(benchmark_case)
            except Exception:  # noqa: BLE001
                LOGGER.exception("case=%s mode=%s snapshot write failed", benchmark_case.case_id, self.mode)
            return EvaluationResult(
                case_id=benchmark_case.case_id,
                mode=self.mode,
                t0_accuracy=None,
                t1_accuracy=None,
                retrieval_precision=0.0,
                autonomy_flag=False,
                status="failed",
                error=str(exc),
            )

    def run_cases(self, cases: Iterable[BenchmarkCase | Dict[str, Any]]) -> List[EvaluationResult]:
        """Evaluate a sequence of cases."""

        results: List[EvaluationResult] = []
        items = [item if isinstance(item, BenchmarkCase) else BenchmarkCase.from_payload(item) for item in cases]
        for case in tqdm(items, desc=f"{self.mode}", unit="case"):
            if not self.config.shared_memory_across_cases:
                self._initialize_case_state(case)
            results.append(self.evaluate_case(case))
        metrics = pd.DataFrame([result.metrics_row() for result in results])
        metrics.to_csv(self.run_dir / f"metrics_{_slugify(self.mode)}.csv", index=False)
        details = [result.snapshot() for result in results]
        (self.run_dir / f"metrics_{_slugify(self.mode)}.json").write_text(json.dumps(details, indent=2, sort_keys=True), encoding="utf-8")
        return results


def evaluate_case(
    case: BenchmarkCase | Dict[str, Any],
    mode: str,
    *,
    output_dir: str | Path | None = None,
    llm_client: JSONChatClient | None = None,
    config: RunnerConfig | None = None,
    semantic_embedding_adapter: Any | None = None,
) -> EvaluationResult:
    """Module-level convenience wrapper required by the benchmark protocol."""

    runner = BenchmarkRunner(
        mode=mode,
        output_dir=output_dir,
        llm_client=llm_client,
        config=config,
        semantic_embedding_adapter=semantic_embedding_adapter,
    )
    return runner.evaluate_case(case)


__all__ = [
    "ALL_MODES",
    "BenchmarkCase",
    "BenchmarkRunner",
    "EvaluationResult",
    "HeuristicBenchmarkClient",
    "MODE_A_FULL",
    "MODE_B_AMNESIC",
    "MODE_C_HASH",
    "MODE_D_LLMONLY",
    "RunnerConfig",
    "evaluate_case",
    "load_cases",
]
