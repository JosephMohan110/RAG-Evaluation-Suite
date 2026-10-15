"""Load and validate labeled evaluation cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    query: str
    expected_answerable: bool
    relevant_sources: tuple[str, ...]
    expected_terms: tuple[str, ...]


def _parse_case(row: Any, line_number: int) -> EvaluationCase:
    if not isinstance(row, dict):
        raise ValueError(f"Line {line_number}: each JSONL record must be an object.")

    case_id = row.get("id")
    query = row.get("query")
    answerable = row.get("expected_answerable")
    sources = row.get("relevant_sources")
    terms = row.get("expected_terms", [])
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"Line {line_number}: 'id' must be a non-empty string.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Line {line_number}: 'query' must be a non-empty string.")
    if not isinstance(answerable, bool):
        raise ValueError(f"Line {line_number}: 'expected_answerable' must be boolean.")
    if not isinstance(sources, list) or any(not isinstance(item, str) or not item for item in sources):
        raise ValueError(f"Line {line_number}: 'relevant_sources' must be a list of non-empty strings.")
    if not isinstance(terms, list) or any(not isinstance(item, str) or not item for item in terms):
        raise ValueError(f"Line {line_number}: 'expected_terms' must be a list of non-empty strings.")
    if answerable and not sources:
        raise ValueError(f"Line {line_number}: answerable cases need at least one relevant source.")
    if not answerable and sources:
        raise ValueError(f"Line {line_number}: unanswerable cases must have no relevant sources.")
    return EvaluationCase(
        case_id=case_id.strip(),
        query=query.strip(),
        expected_answerable=answerable,
        relevant_sources=tuple(sources),
        expected_terms=tuple(terms),
    )


def load_dataset(path: Path) -> tuple[list[EvaluationCase], str]:
    """Read JSONL records and return cases with the source file's SHA-256."""
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError(f"Evaluation dataset is empty: {path}")

    cases: list[EvaluationCase] = []
    ids: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Line {line_number}: invalid JSON: {error.msg}") from error
        case = _parse_case(row, line_number)
        if case.case_id in ids:
            raise ValueError(f"Line {line_number}: duplicate case id {case.case_id!r}.")
        ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"Evaluation dataset contains no cases: {path}")
    return cases, hashlib.sha256(raw).hexdigest()