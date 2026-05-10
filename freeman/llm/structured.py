"""Shared helpers for structured LLM interactions."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

REPAIR_SYSTEM_PROMPT = """You repair Freeman simulation packages using structured verifier feedback.

Return exactly one JSON object with:
- schema: corrected Freeman domain schema
- policies: corrected Freeman Policy snapshots
- assumptions: short list of assumptions

Rules:
- Preserve the domain semantics and actor/resource naming unless the feedback requires otherwise.
- Repair only the fields implicated by the feedback when possible.
- Use the violation details, including field names, observed values, expected bounds, and repair_targets.
- Keep the package compact and numerically stable.
- Use only Freeman-supported evolution types: linear, stock_flow, logistic, threshold, coupled.
- Use the verifier schema spec when provided. A verifier-clean package matters more than preserving a broken field.
- If error history is provided, avoid repeating earlier rejected structures.
- Return JSON only.
"""


def strip_code_fences(text: str) -> str:
    """Remove surrounding Markdown code fences from a model response."""

    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1) if match else stripped


def extract_json_object(text: str, *, provider_name: str) -> dict[str, Any]:
    """Parse one JSON object from model output, with fenced and substring fallback."""

    cleaned = strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{provider_name} response did not contain a valid JSON object.")


def build_repair_messages(
    package: Dict[str, Any],
    violations: List[Dict[str, Any]],
    *,
    domain_description: str = "",
    error_history: List[Dict[str, Any]] | None = None,
    verifier_schema_spec: Dict[str, Any] | None = None,
    repair_stage: str = "standard",
) -> list[dict[str, str]]:
    """Build the shared repair prompt payload for Freeman package fixes."""

    prompt_payload = {
        "domain_description": domain_description,
        "package": package,
        "violations": violations,
        "repair_stage": repair_stage,
        "error_history": error_history or [],
        "verifier_schema_spec": verifier_schema_spec or {},
    }
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
    ]
