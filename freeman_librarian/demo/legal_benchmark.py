"""Helpers for running Freeman librarian on a small Russian legal benchmark subset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
import json
import re
from typing import Any

from freeman_librarian.bootstrap import DocumentBootstrapper
from freeman_librarian.memory.knowledgegraph import KnowledgeGraph

DEFAULT_LEGAL_BENCHMARK_REPO = "parlorsky/legal-rag-benchmark-ru"


@dataclass
class LegalDemoRunResult:
    """Artifacts written by the legal benchmark demo."""

    repo_id: str
    raw_doc_paths: list[Path]
    prepared_doc_paths: list[Path]
    output_root: Path
    summary: dict[str, Any]


def _require_hf() -> tuple[Any, Any]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Legal benchmark demo requires `huggingface_hub`. Install with `pip install \".[demo]\"`."
        ) from exc
    return hf_hub_download, list_repo_files


def download_legal_benchmark_subset(
    output_root: str | Path,
    *,
    repo_id: str = DEFAULT_LEGAL_BENCHMARK_REPO,
    max_docs: int = 4,
) -> tuple[list[Path], Path | None]:
    """Download a small subset of the legal benchmark sequentially."""

    hf_hub_download, list_repo_files = _require_hf()
    root = Path(output_root).resolve()
    raw_dir = root / "raw_docs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    repo_files = sorted(file for file in list_repo_files(repo_id, repo_type="dataset") if file.startswith("sample_docs/"))
    selected = repo_files[: max(int(max_docs), 1)]
    raw_paths: list[Path] = []
    for repo_file in selected:
        cached = Path(hf_hub_download(repo_id, repo_file, repo_type="dataset"))
        destination = raw_dir / Path(repo_file).name
        copyfile(cached, destination)
        raw_paths.append(destination)

    qa_path = None
    try:
        cached = Path(hf_hub_download(repo_id, "test_dataset.json", repo_type="dataset"))
        qa_path = root / "test_dataset.json"
        copyfile(cached, qa_path)
    except Exception:
        qa_path = None

    return raw_paths, qa_path


def infer_document_title(text: str, fallback_name: str) -> str:
    """Infer a compact document title from header text."""

    fallback_title = _normalize_spaces(Path(fallback_name).stem.replace("_", " "))
    if any(token in fallback_title for token in ("Федеральный закон", "Положение", "Инструкция", "Постановление", "Правила")):
        return fallback_title

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:40]:
        if len(line) < 12:
            continue
        if any(token in line for token in ("Федеральный закон", "Положение", "Инструкция", "Правила", "Постановление")):
            return _normalize_spaces(line)
    for line in lines[:10]:
        if 20 <= len(line) <= 180:
            return _normalize_spaces(line)
    return fallback_name


def infer_issuer(text: str, file_name: str, title: str) -> str:
    """Infer the institutional issuer for a legal document."""

    haystack = f"{file_name}\n{title}\n{text[:5000]}"
    patterns = [
        (r"Правительств[ао] РФ|Правительство Российской Федерации", "Government of the Russian Federation"),
        (r"Банк(?:а)? России|ЦБ РФ|Центральн(?:ый|ого) банк", "Bank of Russia"),
        (r"Фонд микрокредитования Иркутской области|МКК «ФМК ИО»", "Irkutsk Microcredit Fund"),
        (r"Федеральный закон", "Federal legislature"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return label
    return "Legal document issuer"


def infer_process(title: str, file_name: str) -> str:
    """Infer the regulated process or subject from title text."""

    file_stem = _normalize_spaces(Path(file_name).stem.replace("_", " "))
    quoted = re.search(r"[\"«](.+?)[\"»]", title)
    if quoted is not None:
        return _truncate(_normalize_spaces(quoted.group(1).strip(" \"»«")), 120)
    if title.startswith(("«", "\"")):
        return _truncate(_normalize_spaces(title.lstrip("«\"").rstrip("»\"")), 120)

    for candidate in (file_stem, title):
        for pattern in (
            r"\bОб утверждении\b\s+.+",
            r"\bОб\b\s+.+",
            r"\bО порядке\b\s+.+",
            r"\bО\b\s+.+",
        ):
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if match is not None:
                return _truncate(_normalize_spaces(match.group(0).strip(" \"»«")), 120)

    return _truncate(file_stem, 120)


def prepare_legal_benchmark_docs(raw_paths: list[str | Path], output_root: str | Path) -> list[Path]:
    """Convert raw legal texts into extraction-friendly structured notes."""

    prepared_root = Path(output_root).resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)
    prepared_paths: list[Path] = []
    for raw_path in raw_paths:
        source = Path(raw_path).resolve()
        text = source.read_text(encoding="utf-8")
        title = infer_document_title(text, source.stem)
        issuer = infer_issuer(text, source.name, title)
        process = infer_process(title, source.name)
        prepared_text = "\n".join(
            [
                f"Department: {issuer}",
                f"Process: {process}",
                f"{issuer} owns {process}.",
                "",
                f"Source title: {title}",
                f"Source file: {source.name}",
                "Source note: derived heuristically from legal benchmark headers.",
            ]
        )
        destination = prepared_root / f"{source.stem}.md"
        destination.write_text(prepared_text, encoding="utf-8")
        prepared_paths.append(destination)
    return prepared_paths


def run_legal_benchmark_demo(
    output_root: str | Path,
    *,
    repo_id: str = DEFAULT_LEGAL_BENCHMARK_REPO,
    max_docs: int = 4,
    domain_id: str = "legal_benchmark_ru_demo",
) -> LegalDemoRunResult:
    """Download a legal subset, prepare structured docs, and run the librarian bootstrap."""

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw_paths, qa_path = download_legal_benchmark_subset(root, repo_id=repo_id, max_docs=max_docs)
    prepared_paths = prepare_legal_benchmark_docs(raw_paths, root / "prepared_docs")

    kg = KnowledgeGraph(json_path=root / "kg_state.json", auto_load=False, auto_save=True)
    result = DocumentBootstrapper().bootstrap_paths(prepared_paths, domain_id=domain_id, knowledge_graph=kg)

    world_path = root / "world.json"
    world_path.write_text(json.dumps(result.world_state.snapshot(), indent=2, sort_keys=True), encoding="utf-8")
    session_path = root / "session.json"
    result.session_log.save(session_path)
    report_path = root / "report.json"
    report_path.write_text(result.verification_report.to_json(), encoding="utf-8")

    summary = {
        "repo_id": repo_id,
        "domain_id": domain_id,
        "raw_doc_count": len(raw_paths),
        "prepared_doc_count": len(prepared_paths),
        "qa_path": str(qa_path) if qa_path is not None else None,
        "world_path": str(world_path),
        "session_path": str(session_path),
        "report_path": str(report_path),
        "kg_path": str(kg.json_path),
        "actor_count": len(result.ontology.actors),
        "resource_count": len(result.ontology.resources),
        "relation_count": len(result.ontology.relations),
        "conflict_count": len(result.ontology.conflicts),
        "verification_passed": bool(result.verification_report.passed),
        "violations": [violation.snapshot() for violation in result.verification_report.violations],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return LegalDemoRunResult(
        repo_id=repo_id,
        raw_doc_paths=raw_paths,
        prepared_doc_paths=prepared_paths,
        output_root=root,
        summary=summary,
    )


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _truncate(value: str, max_chars: int) -> str:
    normalized = _normalize_spaces(value)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


__all__ = [
    "DEFAULT_LEGAL_BENCHMARK_REPO",
    "LegalDemoRunResult",
    "download_legal_benchmark_subset",
    "infer_document_title",
    "infer_issuer",
    "infer_process",
    "prepare_legal_benchmark_docs",
    "run_legal_benchmark_demo",
]
