"""Deterministic agent harness for replay-driven behavioral tests."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.attentionscheduler import AttentionScheduler, AttentionTask, ObligationQueue
from freeman.agent.forecastregistry import ForecastRegistry
from freeman.agent.proactiveemitter import ProactiveEmitter
from freeman.agent.signalingestion import ManualSignalSource, SignalIngestionEngine, SignalMemory
from freeman.game.runner import SimConfig
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.reconciler import Reconciler
from freeman.memory.sessionlog import SessionLog


@dataclass
class CycleResult:
    """Summary of one end-to-end agent cycle."""

    kg_nodes: int
    decisions: int
    dominant_outcome: str | None
    proactive_events: list[dict[str, Any]] = field(default_factory=list)


class AgentHarness:
    """Replay signals through ingestion, scheduling, and analysis in one deterministic cycle."""

    def __init__(
        self,
        schema: dict,
        signals: list[dict],
        *,
        seed: int = 42,
        enable_emitter: bool = False,
    ) -> None:
        self.schema = copy.deepcopy(schema)
        self.signals = copy.deepcopy(signals)
        self.seed = int(seed)
        self.enable_emitter = bool(enable_emitter)
        self._workspace = TemporaryDirectory(prefix="freeman-agent-harness-")
        self.obligation_queue = ObligationQueue()
        self.signal_memory = SignalMemory()
        self.knowledge_graph = KnowledgeGraph(
            json_path=Path(self._workspace.name) / "kg.json",
            auto_load=False,
            auto_save=False,
        )
        self.forecast_registry = ForecastRegistry(
            auto_load=False,
            auto_save=False,
            obligation_queue=self.obligation_queue,
        )
        self.reconciler = Reconciler()
        self.scheduler = AttentionScheduler(
            attention_budget=max(float(len(self.signals)), 1.0),
            ucb_beta=0.0,
            obligation_queue=self.obligation_queue,
        )
        self.ingestion = SignalIngestionEngine()
        self.pipeline = AnalysisPipeline(
            sim_config=SimConfig(
                max_steps=5,
                level2_check_every=1,
                stop_on_hard_level2=False,
                convergence_check_steps=100,
                convergence_epsilon=3.0e-2,
                seed=self.seed,
            ),
            knowledge_graph=self.knowledge_graph,
            reconciler=self.reconciler,
            forecast_registry=self.forecast_registry,
            emitter=ProactiveEmitter() if self.enable_emitter else None,
        )

    def run_cycle(self) -> CycleResult:
        """Run one deterministic analysis cycle over the replay stream."""

        source = ManualSignalSource(self.signals)
        signals = source.fetch()
        signal_by_id = {signal.signal_id: signal for signal in signals}
        triggers = self.ingestion.ingest(
            ManualSignalSource(signals),
            signal_memory=self.signal_memory,
        )
        for trigger in triggers:
            if trigger.mode == "WATCH":
                continue
            signal = signal_by_id[trigger.signal_id]
            self.scheduler.add_task(
                AttentionTask(
                    task_id=trigger.signal_id,
                    description=signal.text,
                    expected_information_gain=max(trigger.classification.severity, 0.1),
                    cost=1.0,
                    anomaly_score=float(signal.metadata.get("anomaly_score", trigger.mahalanobis_score)),
                    semantic_gap=trigger.classification.semantic_gap,
                    metadata={"topic": signal.topic, "mode": trigger.mode},
                )
            )

        decisions = 0
        dominant_outcome = None
        proactive_events: List[dict[str, Any]] = []
        while True:
            decision = self.scheduler.select_task()
            if decision is None:
                break
            decisions += 1
            result = self.pipeline.run(
                copy.deepcopy(self.schema),
                session_log=SessionLog(session_id=f"harness:{decision.task_id}"),
            )
            dominant_outcome = result.dominant_outcome
            proactive_events.extend(asdict(event) for event in result.proactive_events)

        return CycleResult(
            kg_nodes=len(self.knowledge_graph.nodes()),
            decisions=decisions,
            dominant_outcome=dominant_outcome,
            proactive_events=proactive_events,
        )

    def close(self) -> None:
        """Release temporary workspace state created for the harness."""

        workspace = getattr(self, "_workspace", None)
        if workspace is not None:
            workspace.cleanup()
            self._workspace = None

    def __del__(self) -> None:
        self.close()


__all__ = ["AgentHarness", "CycleResult"]
