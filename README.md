# RAG Evaluation Suite

A reproducible benchmark for a local retrieval-augmented QA pipeline. It uses a Sentence Transformers encoder and FAISS; evaluation does not call a hosted LLM or require an API key.

## What this measures

The included answerer is extractive: it returns the highest-scoring retrieved passage with a source citation, or abstains if its similarity score or top-result-versus-runner-up margin is below a configured threshold. It measures retrieval and grounding behavior without conflating those results with a generative model's quality.

The report includes:

- **Source Hit@k**, **MRR@k**, **Precision@k** (denominator is always k), and **Recall@k**, over answerable cases with labeled relevant sources.
- **Answer behavior accuracy**, including answerable and unanswerable cases, with Wilson 95% confidence intervals for proportions.
- **Answerable answer rate**, **no-answer accuracy**, valid citation coverage/IDs, cited-source precision, and literal expected-term coverage.

These metrics do not establish production quality. The included dataset is small and synthetic; report denominators and confidence intervals, expand it with representative independently labeled cases, and tune the abstention threshold on a separate validation set. Expected-term coverage is a transparent string-match check, not a semantic or hallucination judge. Since the demo answerer copies retrieved text, this suite makes no generative faithfulness claim.

## Requirements and setup

Python 3.11+. The first run downloads `sentence-transformers/all-MiniLM-L6-v2` if it is not cached. Set `HF_HUB_OFFLINE=1` to require a local cache.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, activate an existing compatible environment.

## Run

### Streamlit dashboard

Launch the UI from this project folder:

```powershell
python -m streamlit run app.py
```

The dashboard has two tabs:

- **Upload & ask**: upload up to 20 supported files and ask questions about their contents. Supported formats include searchable PDFs and UTF-8 text, Markdown, Pine Script, Python, JavaScript/TypeScript, JSON/JSONL, CSV, LOG, and YAML. The index is built in memory for your Streamlit session; uploaded content is not added to `data/corpus/` or written to disk. Scanned PDFs need OCR before upload. Use **Clear session** to remove the current uploaded index and chat history.
- **Evaluation benchmark**: loads a saved evaluation report on startup. Adjust top-k and abstention thresholds, then click **Run evaluation** to execute the benchmark. Select individual cases to inspect retrieved passages and citations, and download the full JSON report.

Uploads are limited to 10 MB per file, 25 MB total, 100 pages per PDF, and 500,000 extracted characters. Ask-and-answer output is extractive: it quotes the highest-ranked passage with a source/page citation or abstains when the score/margin thresholds are not met. Uploaded files are processed by the local app; do not expose the app publicly unless you add appropriate authentication and deployment controls.

### Command line

```powershell
python -m rag_eval
```

This evaluates `data/eval_cases.jsonl`, loads the checked-in index and metadata from `data/runtime/vector_store/`, and writes a timestamped report to `results/`.

```powershell
python -m rag_eval --top-k 3 --min-score 0.39 --min-margin 0.05 --output results/my-run.json
python -m rag_eval --help
```

The default score and margin thresholds are demo settings, not universal constants. The margin guard addresses ambiguous nearest neighbors but is not a substitute for calibration. Select both thresholds using held-out validation queries.

Run the no-network unit tests with:

```powershell
python -m unittest discover -s tests -v
```

## Dataset format

Each JSONL line is one case:

```json
{"id":"refund-window","query":"How long do I have to ask for a refund?","expected_answerable":true,"relevant_sources":["refund_policy.txt"],"expected_terms":["30 calendar days"]}
```

Answerable cases require one or more `relevant_sources`; unanswerable cases use an empty list. `expected_terms` are optional and evaluated as literal, case-insensitive substrings in the extractive answer. Keep evaluation labels separate from corpus content and do not tune on the final test set.

## Rebuild the vector store

After changing `data/corpus/` or the embedding model:

```powershell
python -m rag_eval.build_index
```

This encodes one document per `.txt` file, normalizes vectors, and creates an inner-product FAISS index with metadata and a manifest. The evaluator validates dimensions and document counts before searching.

## Interpreting results

The JSON records the dataset SHA-256, vector-store fingerprint, model, cutoff, thresholds, aggregate metrics, and per-case sources/scores/answers. The demonstration thresholds were adjusted while investigating this same tiny dataset, so the answer-behavior score is in-sample and is not independent validation. A result such as `6/6` is not a standalone claim: read it with its Wilson interval and dataset size. Do not quote it as an LLM answer-quality score.

## Layout

```text
RAG Evaluation Suite/
  data/corpus/                       # Demo knowledge base
  data/eval_cases.jsonl              # Labeled evaluation set
  data/runtime/vector_store/         # FAISS index, metadata, manifest
  rag_eval/dataset.py                # JSONL validation and hashing
  rag_eval/evaluate.py               # Pipeline and report generation
  rag_eval/metrics.py                # Metric definitions and intervals
  rag_eval/uploads.py                # Upload validation, PDF extraction, and chunking
  rag_eval/build_index.py             # Local index builder
  rag_eval/__main__.py                # CLI
  app.py                              # Streamlit dashboard
  tests/test_eval.py                  # Evaluation unit tests
  tests/test_uploads.py               # Upload parsing and validation tests
  .streamlit/config.toml              # Streamlit upload-size limit
  results/                            # Generated reports
```
