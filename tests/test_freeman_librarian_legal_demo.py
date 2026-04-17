"""Tests for the legal benchmark adapter used by Freeman librarian demos."""

from __future__ import annotations

from freeman_librarian.demo.legal_benchmark import infer_document_title, infer_issuer, infer_process, prepare_legal_benchmark_docs


def test_legal_demo_heuristics_extract_issuer_and_process() -> None:
    text = (
        'Положение Банка России от 28 июня 2017 г. № 590-П "О порядке формирования кредитными организациями '
        'резервов на возможные потери по ссудам"'
    )
    title = infer_document_title(text, "fallback")

    assert "Положение Банка России" in title
    assert infer_issuer(text, "doc.txt", title) == "Bank of Russia"
    assert "порядке формирования" in infer_process(title, "doc.txt").lower()


def test_prepare_legal_benchmark_docs_emits_structured_markdown(tmp_path) -> None:
    raw = tmp_path / "law.txt"
    raw.write_text(
        'Федеральный закон от 02.12.90 N 395-I "О банках и банковской деятельности"',
        encoding="utf-8",
    )

    prepared = prepare_legal_benchmark_docs([raw], tmp_path / "prepared")

    assert len(prepared) == 1
    payload = prepared[0].read_text(encoding="utf-8")
    assert "Department: Federal legislature" in payload
    assert "Process: О банках и банковской деятельности" in payload
    assert "Federal legislature owns О банках и банковской деятельности." in payload
