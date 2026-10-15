from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from rag_eval.dataset import EvaluationCase, load_dataset
from rag_eval.evaluate import _evaluate_case
from rag_eval.metrics import compute_metrics, wilson_interval


class DatasetTests(unittest.TestCase):
    def test_loads_cases_and_hashes_exact_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text(
                '{"id":"known","query":"Where?","expected_answerable":true,'
                '"relevant_sources":["guide.txt"],"expected_terms":["here"]}\n',
                encoding="utf-8",
            )
            cases, digest = load_dataset(path)
            self.assertEqual(cases[0].case_id, "known")
            self.assertEqual(len(digest), 64)

    def test_rejects_duplicate_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            row = {
                "id": "same",
                "query": "Question?",
                "expected_answerable": False,
                "relevant_sources": [],
                "expected_terms": [],
            }
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case id"):
                load_dataset(path)

    def test_rejects_inconsistent_answerability_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text(
                '{"id":"bad","query":"Question?","expected_answerable":true,'
                '"relevant_sources":[]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "need at least one relevant source"):
                load_dataset(path)


class MetricTests(unittest.TestCase):
    def test_wilson_interval_includes_observed_rate(self) -> None:
        interval = wilson_interval(6, 6)
        self.assertIsNotNone(interval)
        self.assertEqual(interval[1], 1.0)
        self.assertLess(interval[0], 1.0)
        self.assertIsNone(wilson_interval(0, 0))

    def test_retrieval_and_abstention_metrics(self) -> None:
        results = [
            {
                "expected_answerable": True,
                "answer_behavior_correct": True,
                "predicted_answerable": True,
                "retrieved_relevant_sources": ["policy.txt"],
                "retrieved_sources": ["policy.txt", "extra.txt"],
                "relevant_sources": ["policy.txt"],
                "cited_sources": ["policy.txt"],
                "answer": "30 calendar days",
                "expected_terms": ["30 calendar days"],
                "citation_ids_valid": True,
            },
            {
                "expected_answerable": False,
                "answer_behavior_correct": True,
                "predicted_answerable": False,
                "retrieved_relevant_sources": [],
                "retrieved_sources": ["extra.txt"],
                "relevant_sources": [],
                "cited_sources": [],
                "answer": "I don't know.",
                "expected_terms": [],
                "citation_ids_valid": True,
            },
        ]
        metrics = compute_metrics(results, top_k=2)
        self.assertEqual(metrics["source_hit_at_k"]["rate"], 1.0)
        self.assertEqual(metrics["mean_source_precision_at_k"], 0.5)
        self.assertEqual(metrics["answer_behavior_accuracy"]["successes"], 2)
        self.assertEqual(metrics["no_answer_accuracy"]["rate"], 1.0)
        self.assertEqual(metrics["mean_expected_term_coverage"], 1.0)
        self.assertEqual(metrics["valid_citation_coverage"]["rate"], 1.0)


class PipelineTests(unittest.TestCase):
    def test_answer_is_extractively_cited_and_low_score_abstains(self) -> None:
        index = faiss.IndexFlatIP(2)
        index.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
        metadata = [{
            "chunk_id": 0,
            "source": "refund_policy.txt",
            "text": "Refunds are available within 30 calendar days.",
        }]

        class FakeModel:
            def __init__(self, vector: list[float]) -> None:
                self.vector = vector

            def encode(self, *args: object, **kwargs: object) -> np.ndarray:
                return np.asarray([self.vector], dtype=np.float32)

        case = EvaluationCase(
            case_id="refund",
            query="When?",
            expected_answerable=True,
            relevant_sources=("refund_policy.txt",),
            expected_terms=("30 calendar days",),
        )
        result = _evaluate_case(
            case, index, metadata, FakeModel([1.0, 0.0]),
            top_k=1, min_score=0.5, min_margin=0.05,
        )
        self.assertTrue(result["answer_behavior_correct"])
        self.assertEqual(result["citation_ids"], ["S1"])
        self.assertEqual(result["expected_term_coverage"], 1.0)

        abstained = _evaluate_case(
            case, index, metadata, FakeModel([0.0, 1.0]),
            top_k=1, min_score=0.5, min_margin=0.05,
        )
        self.assertFalse(abstained["predicted_answerable"])
        self.assertEqual(abstained["citation_ids"], [])


if __name__ == "__main__":
    unittest.main()