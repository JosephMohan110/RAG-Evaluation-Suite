"""Run the local extractive RAG baseline and write an auditable JSON report."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_eval.dataset import EvaluationCase, load_dataset
from rag_eval.metrics import compute_metrics, score_case_terms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "eval_cases.jsonl"
DEFAULT_STORE = ROOT / "data" / "runtime" / "vector_store"
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_MIN_SCORE = 0.39
DEFAULT_MIN_MARGIN = 0.05


def _load_store(store_dir: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    manifest_path = store_dir / "manifest.json"
    metadata_path = store_dir / "metadata.json"
    index_path = store_dir / "index.faiss"
    for path in (manifest_path, metadata_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(f"Vector store file is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Vector-store manifest must be a JSON object: {manifest_path}")
    index = faiss.read_index(str(index_path))
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"Vector metadata must be a non-empty list: {metadata_path}")
    if index.ntotal != len(metadata) or index.ntotal != manifest.get("vector_count"):
        raise ValueError("FAISS index, metadata, and manifest document counts do not match.")
    if index.d != manifest.get("dimension"):
        raise ValueError("FAISS index dimension does not match the vector-store manifest.")
    for position, document in enumerate(metadata):
        if not isinstance(document, dict) or not all(
            isinstance(document.get(field), str) and document[field].strip()
            for field in ("source", "text")
        ):
            raise ValueError(f"Invalid vector metadata record at position {position}.")
    return index, metadata, manifest


def _retrieval(
    index: Any, metadata: list[dict[str, Any]], model: Any, query: str, top_k: int
) -> tuple[list[dict[str, Any]], float | None]:
    vector = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    if vector.shape != (1, index.d):
        raise ValueError(f"Query encoder returned shape {vector.shape}; expected (1, {index.d}).")
    result_count = min(max(top_k, 2), index.ntotal)
    scores, positions = index.search(vector, result_count)
    score_margin = (
        float(scores[0][0] - scores[0][1])
        if result_count > 1 and positions[0][1] >= 0
        else None
    )
    results = []
    for score, position in zip(scores[0][:top_k], positions[0][:top_k]):
        if position < 0:
            continue
        document = metadata[int(position)]
        results.append(
            {
                "source": document["source"],
                "page": document.get("page"),
                "chunk_id": document.get("chunk_id"),
                "score": round(float(score), 6),
                "citation_id": f"S{len(results) + 1}",
                "text": document["text"],
            }
        )
    return results, score_margin


def _extractive_answer(
    retrieved: list[dict[str, Any]],
    min_score: float,
    min_margin: float,
    score_margin: float | None,
) -> tuple[str, list[str], list[str]]:
    if (
        not retrieved
        or retrieved[0]["score"] < min_score
        or (score_margin is not None and score_margin < min_margin)
    ):
        return "I don't know based on the available sources.", [], []
    best = retrieved[0]
    text = re.sub(r"\s+", " ", best["text"]).strip()
    answer = f"{text} [{best['citation_id']}]"
    return answer, [best["citation_id"]], [best["source"]]


def _evaluate_case(
    case: EvaluationCase,
    index: Any,
    metadata: list[dict[str, Any]],
    model: Any,
    top_k: int,
    min_score: float,
    min_margin: float,
) -> dict[str, Any]:
    retrieved, score_margin = _retrieval(index, metadata, model, case.query, top_k)
    answer, citation_ids, cited_sources = _extractive_answer(
        retrieved, min_score, min_margin, score_margin
    )
    predicted_answerable = bool(citation_ids)
    retrieved_sources = [row["source"] for row in retrieved]
    relevant_retrieved = sorted(set(retrieved_sources) & set(case.relevant_sources))
    term_coverage, matched_terms = score_case_terms(answer, case.expected_terms)
    known_citations = {row["citation_id"] for row in retrieved}
    relevant_rank = next(
        (rank for rank, row in enumerate(retrieved, start=1)
         if row["source"] in case.relevant_sources),
        None,
    )
    return {
        "case_id": case.case_id,
        "query": case.query,
        "expected_answerable": case.expected_answerable,
        "predicted_answerable": predicted_answerable,
        "answer_behavior_correct": predicted_answerable == case.expected_answerable,
        "relevant_sources": list(case.relevant_sources),
        "retrieved_sources": retrieved_sources,
        "retrieved_relevant_sources": relevant_retrieved,
        "top_score_margin": round(score_margin, 6) if score_margin is not None else None,
        "retrieval_hit_at_k": bool(relevant_retrieved) if case.expected_answerable else None,
        "reciprocal_rank_at_k": (
            1 / relevant_rank if relevant_rank is not None else 0.0
        ) if case.expected_answerable else None,
        "retrieval_precision_at_k": (
            len(relevant_retrieved) / top_k if case.expected_answerable else None
        ),
        "retrieval_recall_at_k": (
            len(relevant_retrieved) / len(case.relevant_sources)
            if case.expected_answerable else None
        ),
        "valid_citation_coverage": (
            bool(set(cited_sources) & set(case.relevant_sources))
            if case.expected_answerable else None
        ),
        "citation_ids_valid": all(
            citation_id in known_citations for citation_id in citation_ids
        ),
        "cited_sources": cited_sources,
        "cited_source_precision": (
            len(set(cited_sources) & set(case.relevant_sources)) / len(cited_sources)
            if cited_sources and case.expected_answerable else None
        ),
        "expected_terms": list(case.expected_terms),
        "expected_term_coverage": term_coverage,
        "matched_expected_terms": matched_terms,
        "retrieved": retrieved,
        "answer": answer,
        "citation_ids": citation_ids,
    }


def evaluate(
    dataset_path: Path = DEFAULT_DATASET,
    store_dir: Path = DEFAULT_STORE,
    top_k: int = 3,
    min_score: float = DEFAULT_MIN_SCORE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    model_name: str | None = None,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1.")
    if not 0 <= min_margin <= 2:
        raise ValueError("min_margin must be between 0 and 2.")
    cases, dataset_hash = load_dataset(dataset_path)
    index, metadata, manifest = _load_store(store_dir)
    selected_model = model_name or manifest.get("embedding_model")
    if not isinstance(selected_model, str) or not selected_model:
        raise ValueError("No embedding model configured in manifest or CLI.")
    model = SentenceTransformer(selected_model)
    dimension = model.get_sentence_embedding_dimension()
    if dimension != index.d:
        raise ValueError(
            f"Embedding model dimension {dimension} does not match index dimension {index.d}."
        )
    known_sources = {document["source"] for document in metadata}
    unknown_sources = sorted(
        {source for case in cases for source in case.relevant_sources} - known_sources
    )
    if unknown_sources:
        raise ValueError(f"Dataset references sources absent from the vector store: {unknown_sources}")

    results = [
        _evaluate_case(case, index, metadata, model, top_k, min_score, min_margin)
        for case in cases
    ]
    return {
        "schema_version": 1,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "caveats": [
            "The dataset is a small synthetic demonstration, not a representative test set.",
            "The abstention thresholds are illustrative and have not been validated on held-out data.",
            "Answer behavior metrics describe this extractive baseline, not a generative model.",
        ],
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": dataset_hash,
            "case_count": len(cases),
        },
        "vector_store": {
            "manifest": manifest,
            "files_sha256": {
                "index.faiss": hashlib.sha256((store_dir / "index.faiss").read_bytes()).hexdigest(),
                "metadata.json": hashlib.sha256((store_dir / "metadata.json").read_bytes()).hexdigest(),
            },
        },
        "configuration": {
            "top_k": top_k,
            "min_score": min_score,
            "min_margin": min_margin,
            "embedding_model": selected_model,
            "index_metric": manifest.get("index_metric", "inner_product"),
            "answerer": "extractive_top_passage",
        },
        "metrics": compute_metrics(results, top_k),
        "cases": results,
    }