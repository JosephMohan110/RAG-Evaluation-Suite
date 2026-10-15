"""Transparent retrieval, answer-behavior, and citation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import mean
from typing import Any


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if total == 0:
        return None
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total.")
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def proportion(successes: int, total: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else None,
        "wilson_95_ci": wilson_interval(successes, total),
    }


def _term_coverage(answer: str, terms: Sequence[str]) -> tuple[float | None, list[str]]:
    if not terms:
        return None, []
    normalized = " ".join(answer.casefold().split())
    matched = [term for term in terms if " ".join(term.casefold().split()) in normalized]
    return len(matched) / len(terms), matched


def score_case_terms(answer: str, terms: Sequence[str]) -> tuple[float | None, list[str]]:
    """Return literal term coverage and the matching terms for one answer."""
    return _term_coverage(answer, terms)


def compute_metrics(case_results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Aggregate case records. Precision@k always uses k as its denominator."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    answerable = [result for result in case_results if result["expected_answerable"]]
    unanswerable = [result for result in case_results if not result["expected_answerable"]]
    behavior_correct = sum(result["answer_behavior_correct"] for result in case_results)
    source_hits = sum(bool(result["retrieved_relevant_sources"]) for result in answerable)
    citation_ids_valid = sum(result["citation_ids_valid"] for result in case_results)
    answerable_answer_count = sum(result["predicted_answerable"] for result in answerable)
    no_answer_correct = sum(not result["predicted_answerable"] for result in unanswerable)

    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    citation_coverages: list[bool] = []
    term_coverages: list[float] = []
    cited_precisions: list[float] = []
    for result in answerable:
        retrieved_sources = result["retrieved_sources"]
        rank = next(
            (index for index, source in enumerate(retrieved_sources, start=1)
             if source in result["relevant_sources"]),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank is not None else 0.0)
        relevant_retrieved = len(set(result["retrieved_relevant_sources"]))
        precisions.append(relevant_retrieved / top_k)
        recalls.append(relevant_retrieved / len(result["relevant_sources"]))
        citation_coverages.append(
            bool(set(result["cited_sources"]) & set(result["relevant_sources"]))
        )
        coverage, _ = _term_coverage(result["answer"], result["expected_terms"])
        if coverage is not None:
            term_coverages.append(coverage)
        if result["cited_sources"]:
            cited_precisions.append(
                len(set(result["cited_sources"]) & set(result["relevant_sources"]))
                / len(result["cited_sources"])
            )

    return {
        "case_count": len(case_results),
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(unanswerable),
        "source_hit_at_k": proportion(source_hits, len(answerable)),
        "mean_reciprocal_rank_at_k": mean(reciprocal_ranks) if reciprocal_ranks else None,
        "mean_source_precision_at_k": mean(precisions) if precisions else None,
        "mean_source_recall_at_k": mean(recalls) if recalls else None,
        "answer_behavior_accuracy": proportion(behavior_correct, len(case_results)),
        "answerable_answer_rate": proportion(answerable_answer_count, len(answerable)),
        "no_answer_accuracy": proportion(no_answer_correct, len(unanswerable)),
        "valid_citation_coverage": proportion(sum(citation_coverages), len(citation_coverages)),
        "citation_id_validity_rate": proportion(citation_ids_valid, len(case_results)),
        "mean_cited_source_precision": mean(cited_precisions) if cited_precisions else None,
        "mean_expected_term_coverage": mean(term_coverages) if term_coverages else None,
    }