"""Runtime namespace for long-running Freeman agent operation."""

from freeman_librarian.runtime.agent_runtime import AgentRuntime
from freeman_librarian.runtime.checkpoint import CheckpointManager
from freeman_librarian.runtime.event_log import EventLog
from freeman_librarian.runtime.stream import StreamCursorStore


def stream_runtime_main(argv=None):
    """Lazily import the stream runtime entrypoint to avoid package cycles."""

    from freeman_librarian.runtime.stream_runtime import main

    return main(argv)

__all__ = ["AgentRuntime", "CheckpointManager", "EventLog", "StreamCursorStore", "stream_runtime_main"]
