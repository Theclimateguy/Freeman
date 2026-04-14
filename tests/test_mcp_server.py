from __future__ import annotations

from inspect import signature
import sys
from types import ModuleType

from freeman.api.tool_api import FREEMAN_TOOLS
from freeman.mcp.server import _build_tool_callable, build_mcp_server


class _FakeFastMCP:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.args = args
        self.kwargs = kwargs
        self.tools = []

    def add_tool(self, fn, **kwargs) -> None:  # noqa: ANN001
        self.tools.append({"fn": fn, **kwargs})

    def run(self, transport: str = "stdio") -> None:
        self.transport = transport


def test_build_tool_callable_preserves_declared_signature() -> None:
    spec = next(item for item in FREEMAN_TOOLS if item["name"] == "freeman_trace_relation_learning")

    fn = _build_tool_callable(spec)
    params = signature(fn).parameters

    assert list(params) == ["source", "target", "config_path", "relation_type", "last_n_steps"]
    assert params["source"].default is params["source"].empty
    assert params["config_path"].default == "config.yaml"
    assert params["last_n_steps"].default == 10


def test_build_mcp_server_registers_all_tools(monkeypatch) -> None:
    fake_mcp = ModuleType("mcp")
    fake_server = ModuleType("mcp.server")
    fake_fastmcp = ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = _FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)

    server = build_mcp_server(name="Freeman Test")

    assert isinstance(server, _FakeFastMCP)
    assert len(server.tools) == len(FREEMAN_TOOLS)
    assert {tool["name"] for tool in server.tools} == {tool["name"] for tool in FREEMAN_TOOLS}
