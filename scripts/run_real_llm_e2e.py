"""Run real-LLM end-to-end Freeman scenarios with semantic memory recall."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freeman.agent.analysispipeline import AnalysisPipeline, AnalysisPipelineConfig
from freeman.agent.forecastregistry import ForecastRegistry
from freeman.agent.proactiveemitter import ProactiveEmitter
from freeman.agent.signalingestion import ManualSignalSource, ShockClassification, Signal, SignalIngestionEngine, SignalMemory
from freeman.game.runner import SimConfig
from freeman.llm import HashingEmbeddingAdapter, OllamaEmbeddingClient
from freeman.llm.deepseek import DeepSeekChatClient
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.memory.reconciler import Reconciler
from freeman.memory.sessionlog import KGDelta, SessionLog
from freeman.memory.vectorstore import KGVectorStore

MEMORY_PROMPT = """You are producing a fast prior from Freeman long-term memory.

Use only the retrieved memory context below. Do not invent facts that are absent from memory.
Return exactly one JSON object with:
- preliminary_assessment: short paragraph
- remembered_facts: list of 3-5 concrete facts recovered from memory
- predicted_next_events: list of 2-4 near-term predictions
- confidence: float in [0,1]
"""

CLASSIFIER_PROMPT = """Classify this incoming signal for Freeman.

Return exactly one JSON object with:
- shock_type: short label
- severity: float in [0,1]
- semantic_gap: float in [0,1]
- rationale: one short sentence
"""

STATE_INFERENCE_PROMPT = """You are mapping an incoming signal stream into a compact Freeman world template.

All resources are on a 0-100 scale. Use only the provided template resources.
Return exactly one JSON object with:
- overview: short paragraph
- key_mechanisms: list of 3-5 concise mechanism statements
- resource_shocks: object mapping allowed resource ids to additive shocks in [-15, 15]
- dominant_outcome_hypothesis: string
- confidence: float in [0,1]
"""

SIMULATION_INTERPRETATION_PROMPT = """You are interpreting a Freeman simulation for an expert user.

Return exactly one JSON object with:
- executive_summary: short paragraph
- key_dynamics: list of 3-5 concise points
- predicted_next_events: list of 2-4 near-term implications
- caveats: list of 2-4 concise caveats
"""

STOPWORDS = {
    "again",
    "based",
    "case",
    "from",
    "given",
    "have",
    "immediate",
    "memory",
    "only",
    "place",
    "prior",
    "repeat",
    "stored",
    "using",
    "visible",
    "what",
    "which",
}


@dataclass(frozen=True)
class Scenario:
    """One real-LLM evaluation scenario."""

    scenario_id: str
    title: str
    domain_brief: str
    signals: List[Dict[str, Any]]
    follow_up_question: str


def _load_api_key(path: str | Path | None = None) -> str:
    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    key_path = Path(path or REPO_ROOT / "DS.txt").resolve()
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"DeepSeek API key file is empty: {key_path}")
    return key


def _build_scenarios() -> List[Scenario]:
    return [
        Scenario(
            scenario_id="economy_trade_stagflation",
            title="Trade shock and stagflation risk",
            domain_brief=(
                "Model a small open economy hit by a sudden trade-cost shock. "
                "The agent should reason about shipping costs, imported inflation, consumer demand, "
                "central-bank tightening, business investment, and recession risk. "
                "Build a compact, empirically plausible world around the phenomenon and infer which macro outcome "
                "is most likely over the next two quarters."
            ),
            signals=[
                {
                    "signal_id": "econ-1",
                    "source_type": "manual",
                    "topic": "shipping_costs",
                    "text": "Container freight rates jumped roughly 35 percent in three weeks after new tariff announcements and port disruptions.",
                    "entities": ["freight", "tariffs", "ports"],
                    "sentiment": -0.5,
                },
                {
                    "signal_id": "econ-2",
                    "source_type": "manual",
                    "topic": "inflation_expectations",
                    "text": "Household inflation expectations rose while retail foot traffic and discretionary purchases softened.",
                    "entities": ["inflation", "consumption"],
                    "sentiment": -0.4,
                },
                {
                    "signal_id": "econ-3",
                    "source_type": "manual",
                    "topic": "central_bank",
                    "text": "The central bank signaled a hawkish bias because imported goods prices are feeding into headline inflation.",
                    "entities": ["central bank", "inflation"],
                    "sentiment": -0.2,
                },
            ],
            follow_up_question=(
                "We are again seeing tariff headlines and higher shipping costs. Based only on remembered Freeman context, "
                "what immediate prior should we assign to inflation persistence versus outright recession risk?"
            ),
        ),
        Scenario(
            scenario_id="social_relationship_stress",
            title="Relationship stress, trust, and repair",
            domain_brief=(
                "Model a romantic relationship under prolonged work stress. "
                "The agent should reason about communication quality, trust, jealousy, emotional exhaustion, "
                "social support, and the balance between repair and breakup risk. "
                "Build a compact world around this relationship dynamic and infer the most likely short-run path."
            ),
            signals=[
                {
                    "signal_id": "social-1",
                    "source_type": "manual",
                    "topic": "work_stress",
                    "text": "One partner has been working late for six weeks, replies are delayed, and weekend plans were canceled twice.",
                    "entities": ["work stress", "time scarcity"],
                    "sentiment": -0.4,
                },
                {
                    "signal_id": "social-2",
                    "source_type": "manual",
                    "topic": "jealousy",
                    "text": "A public argument was triggered by jealousy around an ex-partner, and trust now feels fragile.",
                    "entities": ["jealousy", "trust"],
                    "sentiment": -0.6,
                },
                {
                    "signal_id": "social-3",
                    "source_type": "manual",
                    "topic": "repair_attempt",
                    "text": "After one honest conversation the tone improved slightly, but both partners still report fatigue and low patience.",
                    "entities": ["repair", "fatigue"],
                    "sentiment": -0.1,
                },
            ],
            follow_up_question=(
                "We are again seeing delayed replies and jealousy after work overload. Based on remembered Freeman context, "
                "what is the immediate prior on repair versus breakup, and which indicators matter first?"
            ),
        ),
        Scenario(
            scenario_id="film_release_buzz",
            title="Film release with strong buzz and mixed reviews",
            domain_brief=(
                "Model the commercial path of a new science-fiction film before release. "
                "The agent should reason about trailer buzz, critic reviews, word of mouth, marketing intensity, "
                "franchise familiarity, opening weekend strength, and box-office legs. "
                "Build a compact world around the release and infer the most likely market outcome over the first month."
            ),
            signals=[
                {
                    "signal_id": "film-1",
                    "source_type": "manual",
                    "topic": "trailer_buzz",
                    "text": "Trailer views and social sharing are unusually strong, especially among younger sci-fi audiences.",
                    "entities": ["trailer", "audience"],
                    "sentiment": 0.6,
                },
                {
                    "signal_id": "film-2",
                    "source_type": "manual",
                    "topic": "critic_reviews",
                    "text": "Critic reactions are mixed: visuals are praised but story coherence is a common complaint.",
                    "entities": ["critics", "reviews"],
                    "sentiment": -0.1,
                },
                {
                    "signal_id": "film-3",
                    "source_type": "manual",
                    "topic": "presales_marketing",
                    "text": "Studio marketing spend increased and pre-sales are solid in urban multiplexes, but family demand is still unclear.",
                    "entities": ["marketing", "pre-sales"],
                    "sentiment": 0.3,
                },
            ],
            follow_up_question=(
                "A sequel now shows the same pattern of high trailer buzz and mixed critics. From remembered Freeman context alone, "
                "what prior should we assign to opening strength versus weak box-office legs?"
            ),
        ),
    ]


def _classify_signal_with_llm(client: DeepSeekChatClient, signal: Signal) -> ShockClassification:
    payload = {
        "signal_id": signal.signal_id,
        "topic": signal.topic,
        "text": signal.text,
        "entities": signal.entities,
        "sentiment": signal.sentiment,
    }
    response = client.chat_json(
        [
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=250,
    )
    return ShockClassification(
        shock_type=str(response.get("shock_type", "routine")),
        severity=float(min(max(response.get("severity", 0.2), 0.0), 1.0)),
        semantic_gap=float(min(max(response.get("semantic_gap", 0.1), 0.0), 1.0)),
        rationale=str(response.get("rationale", "")).strip(),
    )


def _scenario_template(scenario_id: str) -> Dict[str, Any]:
    if scenario_id == "economy_trade_stagflation":
        return {
            "domain_id": scenario_id,
            "name": "Trade Shock Macro Template",
            "description": "Macro template for imported inflation and recession risk.",
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
                        "coupling_weights": {
                            "trade_cost_pressure": 0.11,
                            "policy_rate": -0.09,
                            "business_demand": 0.02,
                        },
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
                    "evolution_params": {
                        "a": 0.88,
                        "c": 2.0,
                        "coupling_weights": {
                            "inflation_pressure": -0.05,
                            "policy_rate": -0.06,
                        },
                    },
                },
                {
                    "id": "policy_rate",
                    "name": "Policy Rate Stance",
                    "value": 28.0,
                    "unit": "index",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "evolution_type": "linear",
                    "evolution_params": {
                        "a": 0.80,
                        "c": 1.0,
                        "coupling_weights": {
                            "inflation_pressure": 0.08,
                            "business_demand": -0.03,
                        },
                    },
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
                        "coupling_weights": {
                            "business_demand": -0.07,
                            "policy_rate": 0.04,
                            "trade_cost_pressure": 0.05,
                        },
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
                    "scoring_weights": {
                        "business_demand": 0.08,
                        "inflation_pressure": -0.06,
                        "recession_risk": -0.08,
                    },
                },
                {
                    "id": "inflation_persistence",
                    "label": "Inflation Persistence",
                    "scoring_weights": {
                        "inflation_pressure": 0.09,
                        "trade_cost_pressure": 0.06,
                        "policy_rate": 0.03,
                        "business_demand": -0.03,
                    },
                },
                {
                    "id": "recession_spiral",
                    "label": "Recession Spiral",
                    "scoring_weights": {
                        "recession_risk": 0.1,
                        "business_demand": -0.08,
                        "policy_rate": 0.03,
                    },
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
    if scenario_id == "social_relationship_stress":
        return {
            "domain_id": scenario_id,
            "name": "Relationship Stress Template",
            "description": "Relationship template for trust, repair, and breakup risk.",
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
                        "coupling_weights": {
                            "work_stress": -0.07,
                            "trust_level": 0.05,
                            "jealousy_pressure": -0.05,
                        },
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
                        "coupling_weights": {
                            "communication_quality": 0.06,
                            "jealousy_pressure": -0.08,
                        },
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
                        "coupling_weights": {
                            "communication_quality": -0.04,
                            "trust_level": -0.05,
                        },
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
                        "coupling_weights": {
                            "communication_quality": 0.05,
                            "work_stress": -0.04,
                        },
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
                        "coupling_weights": {
                            "trust_level": -0.08,
                            "communication_quality": -0.06,
                            "jealousy_pressure": 0.07,
                            "work_stress": 0.04,
                        },
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
                    "scoring_weights": {
                        "repair_capacity": 0.08,
                        "trust_level": 0.07,
                        "communication_quality": 0.06,
                        "breakup_risk": -0.1,
                    },
                },
                {
                    "id": "fragile_stalemate",
                    "label": "Fragile Stalemate",
                    "scoring_weights": {
                        "work_stress": 0.05,
                        "jealousy_pressure": 0.04,
                        "communication_quality": -0.02,
                        "trust_level": -0.02,
                    },
                },
                {
                    "id": "breakup_path",
                    "label": "Breakup Path",
                    "scoring_weights": {
                        "breakup_risk": 0.1,
                        "jealousy_pressure": 0.05,
                        "trust_level": -0.07,
                        "communication_quality": -0.06,
                    },
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
    if scenario_id == "film_release_buzz":
        return {
            "domain_id": scenario_id,
            "name": "Film Release Template",
            "description": "Film release template for buzz, reviews, and box-office dynamics.",
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
                    "evolution_params": {
                        "a": 0.84,
                        "c": 2.0,
                        "coupling_weights": {
                            "critic_sentiment": 0.04,
                            "marketing_intensity": 0.05,
                        },
                    },
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
                    "evolution_params": {
                        "a": 0.82,
                        "c": 1.5,
                        "coupling_weights": {
                            "critic_sentiment": 0.07,
                            "buzz": 0.05,
                        },
                    },
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
                        "coupling_weights": {
                            "buzz": 0.08,
                            "marketing_intensity": 0.06,
                            "critic_sentiment": 0.03,
                        },
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
                        "coupling_weights": {
                            "word_of_mouth": 0.08,
                            "critic_sentiment": 0.05,
                            "opening_weekend": 0.03,
                        },
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
                    "scoring_weights": {
                        "opening_weekend": 0.06,
                        "box_office_legs": 0.08,
                        "word_of_mouth": 0.06,
                        "critic_sentiment": 0.04,
                    },
                },
                {
                    "id": "front_loaded_opening",
                    "label": "Front-Loaded Opening",
                    "scoring_weights": {
                        "opening_weekend": 0.08,
                        "box_office_legs": -0.07,
                        "buzz": 0.05,
                        "critic_sentiment": -0.04,
                    },
                },
                {
                    "id": "underperformer",
                    "label": "Underperformer",
                    "scoring_weights": {
                        "buzz": -0.04,
                        "critic_sentiment": -0.08,
                        "opening_weekend": -0.06,
                        "box_office_legs": -0.08,
                    },
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
    raise KeyError(f"Unknown scenario id: {scenario_id}")


def _resource_catalog(schema: Dict[str, Any]) -> Dict[str, str]:
    return {resource["id"]: resource["name"] for resource in schema["resources"]}


def _infer_domain_state(
    *,
    client: DeepSeekChatClient,
    scenario: Scenario,
    schema: Dict[str, Any],
    signals: List[Signal],
    triggers: List[Any],
) -> Dict[str, Any]:
    payload = {
        "scenario_title": scenario.title,
        "domain_brief": scenario.domain_brief,
        "resources": _resource_catalog(schema),
        "outcomes": [outcome["id"] for outcome in schema["outcomes"]],
        "signal_stream": [
            {
                "topic": signal.topic,
                "text": signal.text,
                "classification": asdict(trigger.classification),
                "mode": trigger.mode,
            }
            for signal, trigger in zip(signals, triggers, strict=False)
        ],
    }
    raw = client.chat_json(
        [
            {"role": "system", "content": STATE_INFERENCE_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.1,
        max_tokens=1200,
    )
    allowed = set(_resource_catalog(schema))
    shocks = {
        resource_id: float(min(max(value, -15.0), 15.0))
        for resource_id, value in raw.get("resource_shocks", {}).items()
        if resource_id in allowed
    }
    return {
        "overview": str(raw.get("overview", "")).strip(),
        "key_mechanisms": [str(item).strip() for item in raw.get("key_mechanisms", []) if str(item).strip()],
        "resource_shocks": shocks,
        "dominant_outcome_hypothesis": str(raw.get("dominant_outcome_hypothesis", "")).strip(),
        "confidence": float(min(max(raw.get("confidence", 0.0), 0.0), 1.0)),
    }


def _apply_resource_shocks(schema: Dict[str, Any], resource_shocks: Dict[str, float]) -> Dict[str, Any]:
    adjusted = json.loads(json.dumps(schema))
    for resource in adjusted["resources"]:
        shock = float(resource_shocks.get(resource["id"], 0.0))
        resource["value"] = float(min(max(resource["value"] + shock, resource["min_value"]), resource["max_value"]))
    return adjusted


def _trajectory_summary(simulation: Dict[str, Any]) -> Dict[str, Any]:
    trajectory = simulation.get("trajectory", [])
    if len(trajectory) < 2:
        return {}
    start = trajectory[0]["resources"]
    end = trajectory[-1]["resources"]
    return {
        resource_id: {
            "start": start[resource_id]["value"],
            "end": end[resource_id]["value"],
            "delta": end[resource_id]["value"] - start[resource_id]["value"],
        }
        for resource_id in start
    }


def _interpret_simulation(
    *,
    client: DeepSeekChatClient,
    scenario: Scenario,
    state_inference: Dict[str, Any],
    pipeline_result: Any,
) -> Dict[str, Any]:
    payload = {
        "scenario_title": scenario.title,
        "state_inference": state_inference,
        "dominant_outcome": pipeline_result.dominant_outcome,
        "final_outcome_probs": pipeline_result.simulation["final_outcome_probs"],
        "confidence": pipeline_result.simulation["confidence"],
        "trajectory_summary": _trajectory_summary(pipeline_result.simulation),
        "verification": pipeline_result.verification,
    }
    raw = client.chat_json(
        [
            {"role": "system", "content": SIMULATION_INTERPRETATION_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.1,
        max_tokens=1000,
    )
    return {
        "executive_summary": str(raw.get("executive_summary", "")).strip(),
        "key_dynamics": [str(item).strip() for item in raw.get("key_dynamics", []) if str(item).strip()],
        "predicted_next_events": [str(item).strip() for item in raw.get("predicted_next_events", []) if str(item).strip()],
        "caveats": [str(item).strip() for item in raw.get("caveats", []) if str(item).strip()],
    }


def _memory_answer(
    *,
    client: DeepSeekChatClient,
    knowledge_graph: KnowledgeGraph,
    question: str,
    top_k: int,
) -> Dict[str, Any]:
    context_nodes = _select_memory_context(knowledge_graph, question, top_k=top_k)
    context_payload = [
        {
            "id": node.id,
            "label": node.label,
            "node_type": node.node_type,
            "content": node.content,
            "metadata": node.metadata,
        }
        for node in context_nodes
    ]
    raw_response = client.chat_json(
        [
            {"role": "system", "content": MEMORY_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "retrieved_context": context_payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.1,
        max_tokens=1000,
    )
    response = {
        "preliminary_assessment": str(raw_response.get("preliminary_assessment", "")).strip(),
        "remembered_facts": [str(item).strip() for item in raw_response.get("remembered_facts", []) if str(item).strip()],
        "predicted_next_events": [
            str(item).strip()
            for item in raw_response.get("predicted_next_events", [])
            if str(item).strip()
        ],
        "confidence": float(min(max(raw_response.get("confidence", 0.0), 0.0), 1.0)),
        "retrieved_node_ids": [node["id"] for node in context_payload],
        "retrieved_node_count": len(context_payload),
        "total_kg_nodes": len(knowledge_graph.nodes()),
    }
    return response


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-zА-Яа-я0-9]+", text.lower())
        if len(token) >= 4 and token not in STOPWORDS
    }


def _select_memory_context(knowledge_graph: KnowledgeGraph, question: str, *, top_k: int) -> List[KGNode]:
    nodes = knowledge_graph.semantic_query(question, top_k=top_k)
    if not nodes:
        return []
    question_tokens = _tokenize(question)
    if not question_tokens:
        return nodes

    scored: List[tuple[int, float, int, KGNode]] = []
    for index, node in enumerate(nodes):
        haystack = " ".join(
            [
                node.id.replace("_", " ").replace(":", " "),
                node.label,
                node.content,
                json.dumps(node.metadata, ensure_ascii=False, sort_keys=True),
            ]
        )
        overlap = len(question_tokens & _tokenize(haystack))
        scored.append((overlap, float(node.confidence), -index, node))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    positive = [node for overlap, _, _, node in scored if overlap > 0]
    if len(positive) >= min(4, len(nodes)):
        return positive
    ranked = [node for _, _, _, node in scored]
    return ranked


def _node_delta(node: KGNode, *, support: int = 1, contradiction: int = 0) -> KGDelta:
    return KGDelta(
        operation="add_node",
        target_id=node.id,
        payload={"node": node.snapshot()},
        support=support,
        contradiction=contradiction,
    )


def _register_signal_nodes(
    *,
    scenario: Scenario,
    knowledge_graph: KnowledgeGraph,
    session_log: SessionLog,
    signals: List[Signal],
    triggers: List[Any],
) -> List[KGNode]:
    trigger_by_id = {trigger.signal_id: trigger for trigger in triggers}
    created: List[KGNode] = []
    for index, signal in enumerate(signals, start=1):
        trigger = trigger_by_id[signal.signal_id]
        node = KGNode(
            id=f"{scenario.scenario_id}:signal:{index}",
            label=f"{scenario.title} signal {index}",
            node_type="signal",
            content=signal.text,
            confidence=max(trigger.classification.severity, 0.25),
            metadata={
                "scenario_id": scenario.scenario_id,
                "topic": signal.topic,
                "shock_type": trigger.classification.shock_type,
                "severity": trigger.classification.severity,
                "semantic_gap": trigger.classification.semantic_gap,
                "rationale": trigger.classification.rationale,
                "mode": trigger.mode,
                "mahalanobis_score": trigger.mahalanobis_score,
            },
        )
        knowledge_graph.add_node(node)
        session_log.add_kg_delta(_node_delta(node))
        created.append(node)
    return created


def _scenario_summary_node(
    *,
    scenario: Scenario,
    state_inference: Dict[str, Any],
    interpretation: Dict[str, Any],
    pipeline_result: Any,
    triggers: List[Any],
) -> KGNode:
    signal_modes = ", ".join(sorted({trigger.mode for trigger in triggers}))
    content = (
        f"Scenario: {scenario.title}. "
        f"LLM overview: {state_inference.get('overview', '')} "
        f"Dominant outcome: {pipeline_result.simulation['final_outcome_probs']}. "
        f"Executive summary: {interpretation.get('executive_summary', '')} "
        f"Key dynamics: {'; '.join(interpretation.get('key_dynamics', []))}. "
        f"Caveats: {'; '.join(interpretation.get('caveats', []))}. "
        f"Trigger modes observed: {signal_modes}. "
        f"Pipeline dominant outcome: {pipeline_result.dominant_outcome}."
    )
    return KGNode(
        id=f"{scenario.scenario_id}:summary",
        label=f"{scenario.title} memory summary",
        node_type="scenario_summary",
        content=content,
        confidence=max(float(pipeline_result.simulation["confidence"]), 0.35),
        metadata={
            "scenario_id": scenario.scenario_id,
            "resource_shocks": state_inference.get("resource_shocks", {}),
            "dominant_outcome_hypothesis": state_inference.get("dominant_outcome_hypothesis"),
            "final_outcome_probs": pipeline_result.simulation["final_outcome_probs"],
            "dominant_outcome": pipeline_result.dominant_outcome,
            "context_node_ids": pipeline_result.metadata.get("context_node_ids", []),
            "forecast_ids": pipeline_result.metadata.get("forecast_ids", []),
            "interpretation": interpretation,
        },
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _render_report(path: Path, results: List[Dict[str, Any]], repeated_probe: Dict[str, Any]) -> Path:
    lines = [
        "# Real LLM End-to-End Evaluation",
        "",
        "This report was generated by `scripts/run_real_llm_e2e.py` using a live DeepSeek chat model for signal interpretation, template shock inference, simulation interpretation, and memory-grounded follow-up answers.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['scenario_id']}",
                "",
                f"- Title: {result['title']}",
                f"- Dominant outcome: {result['pipeline']['dominant_outcome']}",
                f"- Confidence: {result['pipeline']['simulation_confidence']:.4f}",
                f"- Retrieved context during memory answer: {result['follow_up']['retrieved_node_count']} of {result['follow_up']['total_kg_nodes']} total KG nodes",
                f"- Preliminary memory answer: {result['follow_up']['preliminary_assessment']}",
                "",
                "Remembered facts:",
            ]
        )
        for fact in result["follow_up"]["remembered_facts"]:
            lines.append(f"- {fact}")
        lines.extend(["", "Predicted next events:"])
        for event in result["follow_up"]["predicted_next_events"]:
            lines.append(f"- {event}")
        lines.extend(["", ""])

    lines.extend(
        [
            "## Repeated Memory Probe",
            "",
            f"- Question: {repeated_probe['question']}",
            f"- Retrieved context during repeated probe: {repeated_probe['retrieved_node_count']} of {repeated_probe['total_kg_nodes']} total KG nodes",
            f"- Preliminary assessment: {repeated_probe['preliminary_assessment']}",
            "",
            "Remembered facts:",
        ]
    )
    for fact in repeated_probe["remembered_facts"]:
        lines.append(f"- {fact}")
    lines.extend(["", "Predicted next events:"])
    for event in repeated_probe["predicted_next_events"]:
        lines.append(f"- {event}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_evaluation(
    output_dir: Path,
    *,
    max_steps: int,
    seed: int,
    top_k: int,
    scenario_ids: set[str] | None = None,
) -> Dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = DeepSeekChatClient(
        api_key=_load_api_key(),
        timeout_seconds=120.0,
        max_retries=3,
        retry_backoff_seconds=2.0,
    )
    vectorstore = KGVectorStore(path=output_dir / "chroma_db", collection_name="real_llm_eval")
    embedding_model = os.getenv("FREEMAN_EMBEDDING_MODEL", "nomic-embed-text").strip() or "nomic-embed-text"
    embedding_base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip() or "http://127.0.0.1:11434"
    if os.getenv("FREEMAN_USE_HASHING_EMBEDDINGS", "").strip().lower() in {"1", "true", "yes"}:
        embedding_adapter = HashingEmbeddingAdapter()
    else:
        embedding_adapter = OllamaEmbeddingClient(model=embedding_model, base_url=embedding_base_url)
    knowledge_graph = KnowledgeGraph(
        json_path=output_dir / "kg_state.json",
        auto_load=False,
        auto_save=True,
        llm_adapter=embedding_adapter,
        vectorstore=vectorstore,
    )
    forecast_registry = ForecastRegistry(
        json_path=output_dir / "forecasts.json",
        auto_load=False,
        auto_save=True,
    )
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=max_steps, seed=seed),
        knowledge_graph=knowledge_graph,
        reconciler=Reconciler(),
        forecast_registry=forecast_registry,
        emitter=ProactiveEmitter(),
        config=AnalysisPipelineConfig(retrieval_top_k=top_k, max_context_nodes=12),
    )
    ingestion = SignalIngestionEngine()
    signal_memory = SignalMemory()

    results: List[Dict[str, Any]] = []
    scenarios = [scenario for scenario in _build_scenarios() if scenario_ids is None or scenario.scenario_id in scenario_ids]
    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{len(scenarios)}] scenario={scenario.scenario_id}: classify signals", flush=True)
        session_log = SessionLog(session_id=f"real-llm:{scenario.scenario_id}")
        signal_source = ManualSignalSource(scenario.signals)
        signals = signal_source.fetch()
        triggers = ingestion.ingest(
            ManualSignalSource(signals),
            classifier=lambda signal, _client=client: _classify_signal_with_llm(_client, signal),
            signal_memory=signal_memory,
            skip_duplicates_within_hours=0.0,
        )
        signal_nodes = _register_signal_nodes(
            scenario=scenario,
            knowledge_graph=knowledge_graph,
            session_log=session_log,
            signals=signals,
            triggers=triggers,
        )
        schema = _scenario_template(scenario.scenario_id)

        print(f"[{index}/{len(scenarios)}] scenario={scenario.scenario_id}: infer world state", flush=True)
        state_inference = _infer_domain_state(
            client=client,
            scenario=scenario,
            schema=schema,
            signals=signals,
            triggers=triggers,
        )
        state_node = KGNode(
            id=f"{scenario.scenario_id}:state_inference",
            label=f"{scenario.title} state inference",
            node_type="domain_state",
            content=(
                f"{state_inference['overview']} "
                f"Mechanisms: {'; '.join(state_inference['key_mechanisms'])}. "
                f"Outcome hypothesis: {state_inference['dominant_outcome_hypothesis']}."
            ),
            confidence=max(state_inference["confidence"], 0.3),
            metadata={
                "scenario_id": scenario.scenario_id,
                "resource_shocks": state_inference["resource_shocks"],
                "dominant_outcome_hypothesis": state_inference["dominant_outcome_hypothesis"],
            },
        )
        knowledge_graph.add_node(state_node)
        session_log.add_kg_delta(_node_delta(state_node, support=2))

        shocked_schema = _apply_resource_shocks(schema, state_inference["resource_shocks"])
        print(f"[{index}/{len(scenarios)}] scenario={scenario.scenario_id}: simulate template", flush=True)
        pipeline_result = pipeline.run(
            shocked_schema,
            policies=[],
            session_log=session_log,
        )
        interpretation = _interpret_simulation(
            client=client,
            scenario=scenario,
            state_inference=state_inference,
            pipeline_result=pipeline_result,
        )
        summary_node = _scenario_summary_node(
            scenario=scenario,
            state_inference=state_inference,
            interpretation=interpretation,
            pipeline_result=pipeline_result,
            triggers=triggers,
        )
        knowledge_graph.add_node(summary_node)
        knowledge_graph.add_edge(
            KGEdge(
                source=summary_node.id,
                target=state_node.id,
                relation_type="summarizes",
                confidence=0.95,
            )
        )
        for signal_node in signal_nodes:
            knowledge_graph.add_edge(
                KGEdge(
                    source=summary_node.id,
                    target=signal_node.id,
                    relation_type="supported_by",
                    confidence=0.9,
                )
            )
        knowledge_graph.add_edge(
            KGEdge(
                source=summary_node.id,
                target=f"analysis:{pipeline_result.world.domain_id}:{pipeline_result.world.t}",
                relation_type="summarizes",
                confidence=0.95,
            )
        )

        follow_up = _memory_answer(
            client=client,
            knowledge_graph=knowledge_graph,
            question=scenario.follow_up_question,
            top_k=top_k,
        )

        scenario_payload = {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "signals": scenario.signals,
            "signal_triggers": [
                {
                    "signal_id": trigger.signal_id,
                    "mode": trigger.mode,
                    "mahalanobis_score": trigger.mahalanobis_score,
                    "classification": asdict(trigger.classification),
                }
                for trigger in triggers
            ],
            "state_inference": state_inference,
            "pipeline": {
                "dominant_outcome": pipeline_result.dominant_outcome,
                "simulation_confidence": float(pipeline_result.simulation["confidence"]),
                "final_outcome_probs": pipeline_result.simulation["final_outcome_probs"],
                "trajectory_summary": _trajectory_summary(pipeline_result.simulation),
                "context_node_ids": pipeline_result.metadata.get("context_node_ids", []),
                "context_node_count": pipeline_result.metadata.get("context_node_count", 0),
                "proactive_events": [asdict(event) for event in pipeline_result.proactive_events],
            },
            "interpretation": interpretation,
            "follow_up": follow_up,
        }
        _write_json(output_dir / f"{scenario.scenario_id}.json", scenario_payload)
        results.append(scenario_payload)
        print(f"[{index}/{len(scenarios)}] scenario={scenario.scenario_id}: done", flush=True)

    repeated_question = (
        "Repeat the relationship case: delayed replies, jealousy, and work overload are visible again. "
        "Using stored Freeman memory only, what immediate prior should we place on repair versus breakup?"
    )
    if results:
        print("[memory] repeated social probe", flush=True)
        repeated_probe = _memory_answer(
            client=client,
            knowledge_graph=knowledge_graph,
            question=repeated_question,
            top_k=top_k,
        )
        repeated_probe["question"] = repeated_question
        _write_json(output_dir / "repeated_social_probe.json", repeated_probe)
        _render_report(output_dir / "report.md", results, repeated_probe)
        knowledge_graph.export_json(output_dir / "kg_export.json")
        knowledge_graph.export_dot(output_dir / "kg_export.dot")
    else:
        repeated_probe = {
            "question": repeated_question,
            "preliminary_assessment": "",
            "remembered_facts": [],
            "predicted_next_events": [],
            "confidence": 0.0,
            "retrieved_node_ids": [],
            "retrieved_node_count": 0,
            "total_kg_nodes": len(knowledge_graph.nodes()),
        }
    return {
        "output_dir": str(output_dir),
        "scenario_count": len(results),
        "results": results,
        "repeated_probe": repeated_probe,
        "kg_path": str(knowledge_graph.json_path),
        "vectorstore_path": str(vectorstore.path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/real_llm_e2e", help="Directory for generated artifacts.")
    parser.add_argument("--max-steps", type=int, default=12, help="Freeman simulation horizon for each scenario.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic Freeman seed.")
    parser.add_argument("--top-k", type=int, default=6, help="Semantic retrieval top-K before 1-hop expansion.")
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=[],
        help="Optional scenario id filter; may be supplied multiple times.",
    )
    args = parser.parse_args()

    payload = run_evaluation(
        Path(args.output_dir).resolve(),
        max_steps=args.max_steps,
        seed=args.seed,
        top_k=args.top_k,
        scenario_ids=set(args.scenario_id) or None,
    )
    compact = {
        "output_dir": payload["output_dir"],
        "scenario_count": payload["scenario_count"],
        "kg_path": payload["kg_path"],
        "vectorstore_path": payload["vectorstore_path"],
        "scenarios": [
            {
                "scenario_id": item["scenario_id"],
                "dominant_outcome": item["pipeline"]["dominant_outcome"],
                "confidence": item["pipeline"]["simulation_confidence"],
                "retrieved_node_count": item["follow_up"]["retrieved_node_count"],
                "total_kg_nodes": item["follow_up"]["total_kg_nodes"],
            }
            for item in payload["results"]
        ],
        "repeated_probe": {
            "retrieved_node_count": payload["repeated_probe"]["retrieved_node_count"],
            "total_kg_nodes": payload["repeated_probe"]["total_kg_nodes"],
            "confidence": payload["repeated_probe"]["confidence"],
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
