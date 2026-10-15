"""Build a normalized FAISS inner-product index from the text corpus."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "corpus"
DEFAULT_STORE = ROOT / "data" / "runtime" / "vector_store"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_documents(corpus_dir: Path) -> list[dict[str, Any]]:
    documents = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(
                {
                    "chunk_id": len(documents),
                    "source": path.name,
                    "page": None,
                    "chunk_index": 0,
                    "text": text,
                }
            )
    if not documents:
        raise ValueError(f"No non-empty .txt documents found in {corpus_dir}")
    return documents


def build(corpus_dir: Path, store_dir: Path, model_name: str) -> None:
    documents = load_documents(corpus_dir)
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [document["text"] for document in documents],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(documents):
        raise ValueError(f"Encoder returned invalid vector shape: {vectors.shape}")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    store_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(store_dir / "index.faiss"))
    (store_dir / "metadata.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "embedding_model": model_name,
        "dimension": int(vectors.shape[1]),
        "vector_count": len(documents),
        "documents": [document["source"] for document in documents],
        "index_metric": "inner_product_on_l2_normalized_vectors",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (store_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    build(args.corpus, args.store, args.model)
    print(f"Built FAISS index for {args.corpus} at {args.store}")


if __name__ == "__main__":
    main()
