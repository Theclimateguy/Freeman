"""Role-scoped hive-mind runtime for Freeman agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import signal
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml

from freeman.agent.consciousness import ConsciousState, TraceEvent
from freeman.agent.domainregistry import ROLE_TRAIL_SCOPE, trail_scope_for_role
from freeman.core.types import AgentRole, TrailType
from freeman.interface.factory import (
    build_embedding_adapter,
    build_knowledge_graph,
    build_vectorstore,
    resolve_path,
    resolve_runtime_path,
)
from freeman.llm.ollama import OllamaChatClient
from freeman.llm.openai import OpenAIChatClient
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.selfmodel import SelfModelGraph
from freeman.runtime.checkpoint import CheckpointManager
from freeman.runtime.event_log import EventLog
from freeman.runtime.lock_backend import FileSystemLockBackend, InMemoryLockBackend, RedisLockBackend
from freeman.utils import deep_copy_jsonable

LOGGER = logging.getLogger("freeman.runtime.hive")

HIVE_ROLE_ORDER: tuple[AgentRole, ...] = ("ingestor", "repairer", "planner", "narrator", "verifier")
ROLE_OUTPUT_TRAIL: dict[AgentRole, TrailType] = {
    "ingestor": "ingest",
    "repairer": "repair",
    "planner": "read_plan",
    "narrator": "llm_propose",
    "verifier": "verified",
}
DEFAULT_HIVE_CONFIG: dict[str, Any] = {
    "agent_stack": {
        "enabled": True,
        "runtime_id": "hive-local",
        "role_order": list(HIVE_ROLE_ORDER),
        "frontier_limit_per_role": 5,
        "max_actions_per_cycle": 25,
        "max_role_revisits_per_node": 1,
        "lock_backend": "memory",
        "lock_redis_url": "",
        "lock_filesystem_path": "",
        "lock_ttl_seconds": 120.0,
        "checkpoint_path": "",
        "event_log_path": "",
        "llm": {
            "enabled": False,
            "roles": ["narrator", "planner"],
            "temperature": 0.1,
            "max_tokens": 512,
            "role_models": {
                "default": {
                    "provider": "ollama",
                    "model": "qwen2.5-coder:14b",
                    "base_url": "http://127.0.0.1:11434",
                    "timeout_seconds": 120.0,
                },
                "openai_compatible_example": {
                    "provider": "openai-compatible",
                    "model": "gpt-4o-mini",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout_seconds": 90.0,
                },
            },
        },
    },
    "memory": {
        "json_path": "./data/kg_state.json",
        "embedding_provider": "hashing",
        "hashing_embedding_dimension": 384,
        "vector_store": {"enabled": False},
    },
    "runtime": {
        "runtime_path": "./data/runtime",
    },
    "llm": {
        "provider": "ollama",
        "model": "qwen2.5-coder:14b",
        "base_url": "http://127.0.0.1:11434",
        "timeout_seconds": 90.0,
    },
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return deep_copy_jsonable(DEFAULT_HIVE_CONFIG)
    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return _merge_dicts(DEFAULT_HIVE_CONFIG, payload)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value).strip())
    return token.strip("-") or "item"


def _normalize_trail(value: Any) -> str | None:
    if value in {None, "", "None"}:
        return None
    return str(value)


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


@dataclass(frozen=True)
class RoleClientBinding:
    """Resolved chat client for one role."""

    role: AgentRole
    provider: str
    model: str
    base_url: str
    client: Any | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.client is not None and self.error is None


@dataclass(frozen=True)
class HiveRuntimeConfig:
    """Runtime knobs for the role dispatcher."""

    enabled: bool = True
    runtime_id: str = "hive-local"
    role_order: tuple[AgentRole, ...] = HIVE_ROLE_ORDER
    frontier_limit_per_role: int = 5
    max_actions_per_cycle: int = 25
    max_role_revisits_per_node: int = 1
    lock_ttl_seconds: float = 120.0
    llm_enabled: bool = False
    llm_roles: tuple[AgentRole, ...] = ("narrator", "planner")
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HiveRuntimeConfig":
        stack_cfg = dict(config.get("agent_stack", {}) or {})
        llm_cfg = dict(stack_cfg.get("llm", {}) or {})
        role_order = tuple(_coerce_role(role) for role in stack_cfg.get("role_order", HIVE_ROLE_ORDER))
        llm_roles = tuple(_coerce_role(role) for role in llm_cfg.get("roles", ["narrator", "planner"]))
        return cls(
            enabled=bool(stack_cfg.get("enabled", True)),
            runtime_id=str(stack_cfg.get("runtime_id") or "hive-local"),
            role_order=role_order,
            frontier_limit_per_role=max(int(stack_cfg.get("frontier_limit_per_role", 5)), 1),
            max_actions_per_cycle=max(int(stack_cfg.get("max_actions_per_cycle", 25)), 0),
            max_role_revisits_per_node=max(int(stack_cfg.get("max_role_revisits_per_node", 1)), 0),
            lock_ttl_seconds=max(float(stack_cfg.get("lock_ttl_seconds", 120.0)), 0.0),
            llm_enabled=bool(llm_cfg.get("enabled", False)),
            llm_roles=llm_roles,
            llm_temperature=float(llm_cfg.get("temperature", 0.1)),
            llm_max_tokens=max(int(llm_cfg.get("max_tokens", 512)), 1),
        )


@dataclass(frozen=True)
class HiveAction:
    """One role action applied to a KG node."""

    role: AgentRole
    node_id: str
    input_trail: str | None
    output_trail: TrailType
    event_id: str
    llm_used: bool = False
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "node_id": self.node_id,
            "input_trail": self.input_trail,
            "output_trail": self.output_trail,
            "event_id": self.event_id,
            "llm_used": self.llm_used,
            "summary": self.summary,
            "metadata": deep_copy_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class HiveRuntimeReport:
    """Summary of one or more hive runtime cycles."""

    runtime_id: str
    cycles: int
    actions: list[HiveAction]
    skipped: dict[str, int] = field(default_factory=dict)
    checkpoint_path: str = ""
    event_log_path: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "cycles": int(self.cycles),
            "action_count": len(self.actions),
            "actions": [action.snapshot() for action in self.actions],
            "skipped": dict(self.skipped),
            "checkpoint_path": self.checkpoint_path,
            "event_log_path": self.event_log_path,
        }


def _coerce_role(value: Any) -> AgentRole:
    role = str(value).strip()
    if role not in ROLE_TRAIL_SCOPE:
        raise ValueError(f"Unsupported hive role: {value}")
    return role  # type: ignore[return-value]


def _provider_profile_for_role(config: Mapping[str, Any], role: AgentRole) -> dict[str, Any]:
    stack_cfg = dict(config.get("agent_stack", {}) or {})
    stack_llm_cfg = dict(stack_cfg.get("llm", {}) or {})
    role_models = dict(stack_llm_cfg.get("role_models", {}) or {})
    base_llm = dict(config.get("llm", {}) or {})
    default_profile = _merge_dicts(base_llm, dict(role_models.get("default", {}) or {}))
    role_profile = dict(role_models.get(role, {}) or {})
    return _merge_dicts(default_profile, role_profile)


def build_hive_role_clients(config: Mapping[str, Any]) -> dict[AgentRole, RoleClientBinding]:
    """Build role-scoped chat clients without performing network calls."""

    bindings: dict[AgentRole, RoleClientBinding] = {}
    runtime_config = HiveRuntimeConfig.from_config(config)
    for role in runtime_config.role_order:
        profile = _provider_profile_for_role(config, role)
        provider = str(profile.get("provider", "") or "ollama").strip().lower()
        if provider in {"qwen", "local-qwen", "local_qwen"}:
            provider = "ollama"
        model = str(profile.get("model", "") or ("qwen2.5-coder:14b" if provider == "ollama" else "gpt-4o-mini"))
        base_url = str(
            profile.get("base_url", "")
            or ("http://127.0.0.1:11434" if provider == "ollama" else "https://api.openai.com/v1")
        )
        timeout_seconds = float(profile.get("timeout_seconds", 120.0 if provider == "ollama" else 90.0))
        if provider == "ollama":
            bindings[role] = RoleClientBinding(
                role=role,
                provider=provider,
                model=model,
                base_url=base_url,
                client=OllamaChatClient(model=model, base_url=base_url, timeout_seconds=timeout_seconds),
            )
            continue
        if provider in {"openai", "openai-compatible", "openai_compatible"}:
            api_key_env = str(profile.get("api_key_env", "") or "OPENAI_API_KEY")
            api_key = str(profile.get("api_key", "") or os.getenv(api_key_env, "") or os.getenv("LLM_API_KEY", ""))
            if not api_key and provider in {"openai-compatible", "openai_compatible"} and _is_local_base_url(base_url):
                api_key = "EMPTY"
            if not api_key:
                bindings[role] = RoleClientBinding(
                    role=role,
                    provider=provider,
                    model=model,
                    base_url=base_url,
                    error=f"{api_key_env} or LLM_API_KEY is required for {provider}",
                )
                continue
            bindings[role] = RoleClientBinding(
                role=role,
                provider=provider,
                model=model,
                base_url=base_url,
                client=OpenAIChatClient(
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                ),
            )
            continue
        bindings[role] = RoleClientBinding(
            role=role,
            provider=provider,
            model=model,
            base_url=base_url,
            error=f"unsupported provider: {provider}",
        )
    return bindings


class HiveMindRuntime:
    """Deterministic dispatcher for Freeman's fixed hive-mind role chain."""

    def __init__(
        self,
        *,
        state: ConsciousState,
        knowledge_graph: KnowledgeGraph,
        runtime_path: Path,
        config: HiveRuntimeConfig | None = None,
        role_clients: Mapping[AgentRole, RoleClientBinding | Any] | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        event_log: EventLog | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.state = state
        self.knowledge_graph = knowledge_graph
        self.runtime_path = Path(runtime_path).resolve()
        self.config = config or HiveRuntimeConfig()
        self.role_clients = dict(role_clients or {})
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.checkpoint_path = Path(checkpoint_path or self.runtime_path / "hive_checkpoint.json").resolve()
        self.event_log = event_log or EventLog(self.runtime_path / "hive_event_log.jsonl")
        self._skipped: dict[str, int] = {}
        self.stop_requested = False

    def request_stop(self) -> None:
        """Ask the runtime to stop after the current action/cycle boundary."""

        self.stop_requested = True

    def run(self, *, cycles: int = 1) -> HiveRuntimeReport:
        """Run one or more deterministic role-dispatch cycles."""

        actions: list[HiveAction] = []
        cycles = max(int(cycles), 0)
        cycles_completed = 0
        for _ in range(cycles):
            if self.stop_requested:
                break
            actions.extend(self.run_cycle(remaining_actions=self.config.max_actions_per_cycle))
            cycles_completed += 1
        self._checkpoint()
        return HiveRuntimeReport(
            runtime_id=self.config.runtime_id,
            cycles=cycles_completed,
            actions=actions,
            skipped=dict(self._skipped),
            checkpoint_path=str(self.checkpoint_path),
            event_log_path=str(self.event_log.path),
        )

    def run_cycle(self, *, remaining_actions: int | None = None) -> list[HiveAction]:
        """Run each configured role once against its current frontier."""

        if not self.config.enabled:
            self._skip("runtime_disabled")
            return []
        max_actions = self.config.max_actions_per_cycle if remaining_actions is None else max(int(remaining_actions), 0)
        actions: list[HiveAction] = []
        for role in self.config.role_order:
            if self.stop_requested:
                break
            if len(actions) >= max_actions:
                break
            frontier = self.frontier_for_role(role)
            for node in frontier:
                if self.stop_requested:
                    break
                if len(actions) >= max_actions:
                    break
                action = self._process_node(role, node)
                if action is not None:
                    actions.append(action)
        return actions

    def frontier_for_role(self, role: AgentRole) -> list[KGNode]:
        """Return the sorted eligible node frontier for one role."""

        scope = {_normalize_trail(value) for value in trail_scope_for_role(role)}
        candidates: list[KGNode] = []
        for node in self.knowledge_graph.nodes(lazy_embed=False):
            if node.status == "archived":
                continue
            node_trail = _normalize_trail(node.metadata.get("trail_type"))
            if node_trail not in scope:
                continue
            if self._role_count(node, role) >= self.config.max_role_revisits_per_node:
                self._skip("role_revisit_limit")
                continue
            candidates.append(node)
        candidates.sort(
            key=lambda item: (
                -float(item.metadata.get("trail_intensity", 0.0) or 0.0),
                -float(item.confidence),
                str(item.updated_at),
                item.id,
            )
        )
        return candidates[: self.config.frontier_limit_per_role]

    def _process_node(self, role: AgentRole, node: KGNode) -> HiveAction | None:
        agent_id = f"{self.config.runtime_id}:{role}"
        if not self.knowledge_graph.try_lock(node.id, agent_id, lock_ttl_seconds=self.config.lock_ttl_seconds):
            self._skip("lock_conflict")
            return None
        try:
            fresh_node = self.knowledge_graph.get_node(node.id, lazy_embed=False)
            if fresh_node is None:
                self._skip("missing_node")
                return None
            if self._role_count(fresh_node, role) >= self.config.max_role_revisits_per_node:
                self._skip("role_revisit_limit")
                return None
            event_id = self._event_id(role, fresh_node)
            action = self._build_action(role, fresh_node, event_id)
            self._apply_action(fresh_node, action)
            return action
        finally:
            self.knowledge_graph.unlock(node.id, agent_id)

    def _build_action(self, role: AgentRole, node: KGNode, event_id: str) -> HiveAction:
        output_trail = ROLE_OUTPUT_TRAIL[role]
        input_trail = _normalize_trail(node.metadata.get("trail_type"))
        binding = self._binding_for_role(role)
        llm_text = ""
        llm_used = False
        llm_error = None
        if self.config.llm_enabled and role in self.config.llm_roles and binding is not None:
            if binding.available:
                try:
                    llm_text = str(
                        binding.client.chat_text(
                            self._role_messages(role, node),
                            temperature=self.config.llm_temperature,
                            max_tokens=self.config.llm_max_tokens,
                        )
                    ).strip()
                    llm_used = bool(llm_text)
                except Exception as exc:
                    llm_error = str(exc)
            else:
                llm_error = binding.error or "role client unavailable"
        summary = llm_text or self._deterministic_summary(role, node, input_trail, output_trail)
        metadata = {
            "llm_provider": binding.provider if binding is not None else None,
            "llm_model": binding.model if binding is not None else None,
            "llm_error": llm_error,
        }
        return HiveAction(
            role=role,
            node_id=node.id,
            input_trail=input_trail,
            output_trail=output_trail,
            event_id=event_id,
            llm_used=llm_used,
            summary=summary,
            metadata=metadata,
        )

    def _apply_action(self, node: KGNode, action: HiveAction) -> None:
        previous_role = self.state.agent_role
        timestamp = _utc_now()
        metadata = deep_copy_jsonable(node.metadata)
        runtime_meta = dict(metadata.get("hive_runtime", {}) or {})
        role_counts = {str(key): int(value) for key, value in dict(runtime_meta.get("role_counts", {}) or {}).items()}
        role_counts[action.role] = role_counts.get(action.role, 0) + 1
        runtime_meta.update(
            {
                "runtime_id": self.config.runtime_id,
                "last_role": action.role,
                "last_role_at": timestamp.isoformat(),
                "last_event_id": action.event_id,
                "role_counts": role_counts,
            }
        )
        role_outputs = dict(metadata.get("hive_role_outputs", {}) or {})
        role_outputs[action.role] = {
            "summary": action.summary,
            "llm_used": action.llm_used,
            "provider": action.metadata.get("llm_provider"),
            "model": action.metadata.get("llm_model"),
            "error": action.metadata.get("llm_error"),
        }
        history = list(metadata.get("hive_history", []) or [])
        history.append(
            {
                "event_id": action.event_id,
                "role": action.role,
                "input_trail": action.input_trail,
                "output_trail": action.output_trail,
                "timestamp": timestamp.isoformat(),
                "llm_used": action.llm_used,
            }
        )
        metadata.update(
            {
                "trail_type": action.output_trail,
                "trail_intensity": max(float(metadata.get("trail_intensity", 0.0) or 0.0), 1.0),
                "hive_runtime": runtime_meta,
                "hive_role_outputs": role_outputs,
                "hive_history": history[-50:],
            }
        )
        updated_node = KGNode.from_snapshot(node.snapshot())
        updated_node.metadata = metadata

        try:
            self.state.agent_role = action.role
            self.knowledge_graph.update_node(updated_node)
            event = self._trace_event(action, timestamp=timestamp)
            self.state.trace_state.append(event)
            self.event_log.append(event)
            self.state.runtime_metadata["hive_last_event_id"] = action.event_id
            self.state.runtime_metadata["hive_last_role"] = action.role
            self.state.runtime_metadata["hive_action_count"] = int(self.state.runtime_metadata.get("hive_action_count", 0)) + 1
        finally:
            self.state.agent_role = previous_role

    def _trace_event(self, action: HiveAction, *, timestamp: datetime) -> TraceEvent:
        index = len(self.state.trace_state)
        return TraceEvent(
            event_id=action.event_id,
            timestamp=timestamp,
            transition_type="internal",
            trigger_type="manual",
            operator=f"hive_role:{action.role}",
            pre_state_ref=f"hive:{index}",
            post_state_ref=f"hive:{index + 1}",
            input_refs=[f"kg:{action.node_id}"],
            diff={
                "node_id": action.node_id,
                "role": action.role,
                "input_trail": action.input_trail,
                "output_trail": action.output_trail,
                "llm_used": action.llm_used,
            },
            rationale=action.summary,
        )

    def _role_messages(self, role: AgentRole, node: KGNode) -> list[dict[str, str]]:
        payload = {
            "role": role,
            "node": {
                "id": node.id,
                "label": node.label,
                "node_type": node.node_type,
                "content": node.content,
                "confidence": node.confidence,
                "trail_type": _normalize_trail(node.metadata.get("trail_type")),
                "metadata": {
                    key: value
                    for key, value in node.metadata.items()
                    if key not in {"embedding", "hive_history"}
                },
            },
            "output_trail": ROLE_OUTPUT_TRAIL[role],
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a role-scoped Freeman hive-mind worker. "
                    "Do not change the role chain. Return one concise, verifiable handoff note."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]

    def _deterministic_summary(
        self,
        role: AgentRole,
        node: KGNode,
        input_trail: str | None,
        output_trail: TrailType,
    ) -> str:
        source = input_trail if input_trail is not None else "raw"
        return f"{role} routed node {node.id} from {source} to {output_trail}"

    def _binding_for_role(self, role: AgentRole) -> RoleClientBinding | None:
        binding = self.role_clients.get(role)
        if binding is None:
            return None
        if isinstance(binding, RoleClientBinding):
            return binding
        return RoleClientBinding(
            role=role,
            provider=type(binding).__name__,
            model=str(getattr(binding, "model", "")),
            base_url=str(getattr(binding, "base_url", "")),
            client=binding,
        )

    def _role_count(self, node: KGNode, role: AgentRole) -> int:
        runtime_meta = dict(node.metadata.get("hive_runtime", {}) or {})
        role_counts = dict(runtime_meta.get("role_counts", {}) or {})
        return int(role_counts.get(role, 0) or 0)

    def _event_id(self, role: AgentRole, node: KGNode) -> str:
        index = len(self.state.trace_state) + 1
        return f"trace:hive:{_safe_token(self.config.runtime_id)}:{role}:{_safe_token(node.id)}:{index}"

    def _checkpoint(self) -> None:
        self.runtime_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager.save(self.state, self.checkpoint_path)
        if self.knowledge_graph.auto_save:
            self.knowledge_graph.save()

    def _skip(self, reason: str) -> None:
        self._skipped[reason] = self._skipped.get(reason, 0) + 1


def _checkpoint_path(config: Mapping[str, Any], *, config_path: Path, runtime_path: Path) -> Path:
    stack_cfg = dict(config.get("agent_stack", {}) or {})
    configured = str(stack_cfg.get("checkpoint_path", "") or "")
    if configured:
        return resolve_path(config_path.parent, configured, configured)
    return runtime_path / "hive_checkpoint.json"


def _event_log_path(config: Mapping[str, Any], *, config_path: Path, runtime_path: Path) -> Path:
    stack_cfg = dict(config.get("agent_stack", {}) or {})
    configured = str(stack_cfg.get("event_log_path", "") or "")
    if configured:
        return resolve_path(config_path.parent, configured, configured)
    return runtime_path / "hive_event_log.jsonl"


def build_lock_backend_from_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    runtime_path: Path,
) -> Any:
    """Build the configured node-lock backend for hive runtimes."""

    stack_cfg = dict(config.get("agent_stack", {}) or {})
    backend = str(stack_cfg.get("lock_backend", "memory") or "memory").strip().lower()
    if backend in {"", "memory", "inmemory", "in_memory"}:
        return InMemoryLockBackend()
    if backend in {"filesystem", "file", "fs"}:
        configured_path = str(stack_cfg.get("lock_filesystem_path", "") or "")
        lock_dir = (
            resolve_path(config_path.parent, configured_path, configured_path)
            if configured_path
            else runtime_path / "node_locks"
        )
        return FileSystemLockBackend(lock_dir=lock_dir)
    if backend == "redis":
        redis_url = str(stack_cfg.get("lock_redis_url", "") or os.getenv("FREEMAN_REDIS_URL", "")).strip()
        if not redis_url:
            raise RuntimeError("agent_stack.lock_redis_url or FREEMAN_REDIS_URL is required when lock_backend=redis")
        return RedisLockBackend(redis_url=redis_url)
    raise ValueError(f"Unsupported agent_stack.lock_backend: {backend}")


def build_hive_runtime_from_config(
    config_path: str | Path = "config.yaml",
    *,
    resume: bool = True,
) -> HiveMindRuntime:
    """Build a hive runtime from repository config."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    config = _load_yaml_config(resolved_config_path)
    runtime_path = resolve_runtime_path(config, config_path=resolved_config_path)
    runtime_path.mkdir(parents=True, exist_ok=True)
    vectorstore = build_vectorstore(config, config_path=resolved_config_path)
    embedding_adapter, _embedding_label = build_embedding_adapter(config)
    lock_backend = build_lock_backend_from_config(
        config,
        config_path=resolved_config_path,
        runtime_path=runtime_path,
    )
    knowledge_graph = build_knowledge_graph(
        config,
        config_path=resolved_config_path,
        embedding_adapter=embedding_adapter,
        vectorstore=vectorstore,
        auto_load=True,
        auto_save=True,
        lock_backend=lock_backend,
    )
    checkpoint_path = _checkpoint_path(config, config_path=resolved_config_path, runtime_path=runtime_path)
    if resume and checkpoint_path.exists():
        state = CheckpointManager().load(checkpoint_path)
        state.self_model_ref = SelfModelGraph(knowledge_graph)
    else:
        state = ConsciousState(
            world_ref="world:hive:0",
            self_model_ref=SelfModelGraph(knowledge_graph),
            agent_role="planner",
            runtime_metadata={"schema_version": 1},
        )
    runtime_config = HiveRuntimeConfig.from_config(config)
    role_clients = build_hive_role_clients(config)
    return HiveMindRuntime(
        state=state,
        knowledge_graph=knowledge_graph,
        runtime_path=runtime_path,
        config=runtime_config,
        role_clients=role_clients,
        event_log=EventLog(_event_log_path(config, config_path=resolved_config_path, runtime_path=runtime_path)),
        checkpoint_path=checkpoint_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Freeman hive-mind role dispatcher.")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-actions", type=int, default=None)
    return parser


def _install_stop_handlers(runtime: HiveMindRuntime) -> tuple[Any, Any]:
    def _request_stop(signum, frame):  # noqa: ANN001
        del frame
        LOGGER.info("Signal %s received, finishing current hive action/cycle", signum)
        runtime.request_stop()

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    return previous_sigint, previous_sigterm


def _restore_stop_handlers(previous_sigint: Any, previous_sigterm: Any) -> None:
    signal.signal(signal.SIGINT, previous_sigint)
    signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    runtime = build_hive_runtime_from_config(args.config_path, resume=bool(args.resume))
    if args.max_actions is not None:
        runtime.config = HiveRuntimeConfig(
            enabled=runtime.config.enabled,
            runtime_id=runtime.config.runtime_id,
            role_order=runtime.config.role_order,
            frontier_limit_per_role=runtime.config.frontier_limit_per_role,
            max_actions_per_cycle=max(int(args.max_actions), 0),
            max_role_revisits_per_node=runtime.config.max_role_revisits_per_node,
            lock_ttl_seconds=runtime.config.lock_ttl_seconds,
            llm_enabled=runtime.config.llm_enabled,
            llm_roles=runtime.config.llm_roles,
            llm_temperature=runtime.config.llm_temperature,
            llm_max_tokens=runtime.config.llm_max_tokens,
        )
    previous_sigint, previous_sigterm = _install_stop_handlers(runtime)
    try:
        report = runtime.run(cycles=args.cycles)
    finally:
        _restore_stop_handlers(previous_sigint, previous_sigterm)
    print(json.dumps(report.snapshot(), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "HIVE_ROLE_ORDER",
    "ROLE_OUTPUT_TRAIL",
    "HiveAction",
    "HiveMindRuntime",
    "HiveRuntimeConfig",
    "HiveRuntimeReport",
    "RoleClientBinding",
    "build_hive_role_clients",
    "build_hive_runtime_from_config",
    "build_lock_backend_from_config",
    "main",
]
