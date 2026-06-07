"""Structured logging helpers for Freeman runtimes."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import sys
import uuid
from typing import Any

_RUN_ID: ContextVar[str | None] = ContextVar("freeman_run_id", default=None)
_CORRELATION_ID: ContextVar[str | None] = ContextVar("freeman_correlation_id", default=None)


class JsonFormatter(logging.Formatter):
    """Emit compact JSON logs with runtime correlation metadata."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).replace(microsecond=0).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": getattr(record, "run_id", None) or _RUN_ID.get(),
            "correlation_id": getattr(record, "correlation_id", None) or _CORRELATION_ID.get(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def new_run_id() -> str:
    """Return a fresh runtime correlation id."""

    return str(uuid.uuid4())


def set_log_context(*, run_id: str | None = None, correlation_id: str | None = None) -> None:
    """Set context variables used by ``JsonFormatter``."""

    if run_id is not None:
        _RUN_ID.set(str(run_id))
    if correlation_id is not None:
        _CORRELATION_ID.set(str(correlation_id))


def configure_logging(
    *,
    level: str | int = "INFO",
    run_id: str | None = None,
    json_logs: bool = True,
    json_mode: bool | None = None,
    force: bool = False,
) -> str | None:
    """Configure root logging and return the active run id."""

    if json_mode is not None:
        json_logs = bool(json_mode)
    active_run_id = run_id or new_run_id()
    set_log_context(run_id=active_run_id)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), level),
        handlers=[handler],
        force=force,
    )
    return active_run_id


def get_logger(name: str, *, run_id: str | None = None, correlation_id: str | None = None) -> logging.Logger:
    """Return a logger after optionally updating structured context."""

    set_log_context(run_id=run_id, correlation_id=correlation_id)
    return logging.getLogger(name)


__all__ = ["JsonFormatter", "configure_logging", "get_logger", "new_run_id", "set_log_context"]
