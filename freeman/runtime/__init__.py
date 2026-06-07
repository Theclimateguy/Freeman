"""Runtime namespace for long-running Freeman agent operation."""

from freeman.runtime.agent_runtime import AgentRuntime
from freeman.runtime.checkpoint import CheckpointManager
from freeman.runtime.contracts import ConsciousStateContract, KnowledgeGraphContract, WorldStateContract
from freeman.runtime.event_log import EventLog
from freeman.runtime.health import HealthState, get_health, health_from_config
from freeman.runtime.metrics import MetricSample, collect_metrics, metrics_from_config, render_prometheus
from freeman.runtime.stream import StreamCursorStore


def stream_runtime_main(argv=None):
    """Lazily import the stream runtime entrypoint to avoid package cycles."""

    from freeman.runtime.stream_runtime import main

    return main(argv)


def __getattr__(name):
    if name in {"HiveMindRuntime", "HiveRuntimeConfig", "build_hive_runtime_from_config"}:
        from freeman.runtime.hive_runtime import HiveMindRuntime, HiveRuntimeConfig, build_hive_runtime_from_config

        exports = {
            "HiveMindRuntime": HiveMindRuntime,
            "HiveRuntimeConfig": HiveRuntimeConfig,
            "build_hive_runtime_from_config": build_hive_runtime_from_config,
        }
        return exports[name]
    raise AttributeError(name)


__all__ = [
    "AgentRuntime",
    "CheckpointManager",
    "ConsciousStateContract",
    "EventLog",
    "HiveMindRuntime",
    "HiveRuntimeConfig",
    "HealthState",
    "KnowledgeGraphContract",
    "MetricSample",
    "StreamCursorStore",
    "WorldStateContract",
    "build_hive_runtime_from_config",
    "collect_metrics",
    "get_health",
    "health_from_config",
    "metrics_from_config",
    "render_prometheus",
    "stream_runtime_main",
]
