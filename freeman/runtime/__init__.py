"""Runtime namespace for long-running Freeman agent operation."""

from freeman.runtime.agent_runtime import AgentRuntime
from freeman.runtime.checkpoint import CheckpointManager
from freeman.runtime.event_log import EventLog
from freeman.runtime.stream import StreamCursorStore


def stream_runtime_main(argv=None):
    """Lazily import the stream runtime entrypoint to avoid package cycles."""

    from freeman.runtime.stream_runtime import main

    return main(argv)

__all__ = ["AgentRuntime", "CheckpointManager", "EventLog", "StreamCursorStore", "stream_runtime_main"]
