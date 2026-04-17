"""MCP server exposing Freeman tools."""

from __future__ import annotations

import argparse
from inspect import Parameter, Signature
from typing import Any

from freeman_librarian.api.tool_api import FREEMAN_TOOLS, invoke_tool


def _python_type(json_schema: dict[str, Any]) -> type[Any]:
    schema_type = str(json_schema.get("type", "string"))
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list
    if schema_type == "object":
        return dict
    return str


def _build_tool_callable(spec: dict[str, Any]) -> Any:
    name = str(spec["name"])
    parameters_spec = dict(spec.get("parameters", {}))
    properties = dict(parameters_spec.get("properties", {}))
    required = set(parameters_spec.get("required", []))
    ordered_names = [
        *[name for name in properties if name in required],
        *[name for name in properties if name not in required],
    ]

    def _tool(**kwargs: Any) -> Any:
        return invoke_tool(name, kwargs)

    signature_parameters: list[Parameter] = []
    annotations: dict[str, Any] = {"return": Any}
    for param_name in ordered_names:
        schema = properties[param_name]
        annotation = _python_type(dict(schema))
        annotations[param_name] = annotation
        if param_name in required and "default" not in schema:
            default = Parameter.empty
        else:
            default = schema.get("default", None)
        signature_parameters.append(
            Parameter(
                str(param_name),
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
                default=default,
            )
        )

    _tool.__name__ = name
    _tool.__qualname__ = name
    _tool.__doc__ = str(spec.get("description", "")).strip() or name
    _tool.__annotations__ = annotations
    _tool.__signature__ = Signature(parameters=signature_parameters, return_annotation=Any)
    return _tool


def build_mcp_server(
    *,
    name: str = "Freeman",
    instructions: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Any:
    """Build an MCP server that exposes all Freeman tools."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The optional 'mcp' dependency is required. Install with: pip install '.[mcp]'"
        ) from exc

    server = FastMCP(
        name=name,
        instructions=instructions
        or "Freeman is a stateful world-model daemon. Use these tools to inspect its runtime KG, forecasts, anomalies, and relation-learning history.",
        host=host,
        port=port,
    )
    for spec in FREEMAN_TOOLS:
        server.add_tool(
            _build_tool_callable(spec),
            name=str(spec["name"]),
            description=str(spec.get("description", "")).strip(),
        )
    return server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Freeman as an MCP server.")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", default="Freeman")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Freeman MCP server."""

    args = _build_parser().parse_args(argv)
    server = build_mcp_server(name=args.name, host=args.host, port=args.port)
    server.run(transport=str(args.transport))
    return 0


__all__ = ["build_mcp_server", "main"]
