"""Command-line entry point: python -m rag_eval."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_eval.evaluate import (
    DEFAULT_DATASET,
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SCORE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STORE,
    evaluate,
)


def _format_proportion(value: dict[str, Any]) -> str:
    rate = value["rate"]
    if rate is None:
        return "n/a (0 cases)"
    low, high = value["wilson_95_ci"]
    return f"{value['successes']}/{value['total']} ({rate:.1%}; Wilson 95% CI {low:.1%}-{high:.1%})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local extractive RAG baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--model", default=None, help="Override the embedding model in the store manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: timestamped results file).",
    )
    args = parser.parse_args()
    report = evaluate(
        dataset_path=args.dataset,
        store_dir=args.store,
        top_k=args.top_k,
        min_score=args.min_score,
        min_margin=args.min_margin,
        model_name=args.model,
    )
    output_path = args.output
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_OUTPUT_DIR / f"evaluation-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics = report["metrics"]
    print(f"RAG evaluation: {report['dataset']['case_count']} cases")
    print(f"Dataset SHA-256: {report['dataset']['sha256']}")
    print(f"Embedding model: {report['configuration']['embedding_model']}")
    print(
        f"Top-k / min score / min margin: "
        f"{args.top_k} / {args.min_score:.3f} / {args.min_margin:.3f}"
    )
    print(f"Source Hit@k: {_format_proportion(metrics['source_hit_at_k'])}")
    precision = metrics["mean_source_precision_at_k"]
    recall = metrics["mean_source_recall_at_k"]
    print(f"Mean Precision@k: {precision:.3f}" if precision is not None else "Mean Precision@k: n/a")
    print(f"Mean Recall@k: {recall:.3f}" if recall is not None else "Mean Recall@k: n/a")
    print(f"Answer behavior accuracy: {_format_proportion(metrics['answer_behavior_accuracy'])}")
    print(f"No-answer accuracy: {_format_proportion(metrics['no_answer_accuracy'])}")
    mrr = metrics["mean_reciprocal_rank_at_k"]
    print(f"Mean MRR@k: {mrr:.3f}" if mrr is not None else "Mean MRR@k: n/a")
    coverage = metrics["mean_expected_term_coverage"]
    print(
        f"Mean expected-term coverage: {coverage:.1%}"
        if coverage is not None else "Mean expected-term coverage: n/a"
    )
    print(f"Report: {output_path.resolve()}")


if __name__ == "__main__":
    main()