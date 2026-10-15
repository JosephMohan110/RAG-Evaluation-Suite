"""Streamlit dashboard for running and reviewing the local RAG evaluation."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from rag_eval.evaluate import (
    DEFAULT_DATASET,
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SCORE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STORE,
    evaluate,
)
from rag_eval.uploads import SUPPORTED_EXTENSIONS, chunk_uploaded_files

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def _get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _uploads_fingerprint(files: list[Any]) -> str:
    digest = hashlib.sha256()
    for uploaded_file in files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(uploaded_file.getvalue())
        digest.update(b"\0")
    return digest.hexdigest()


def _build_upload_index(files: list[Any]) -> tuple[Any, list[dict[str, Any]]]:
    chunks = chunk_uploaded_files(
        [(uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in files]
    )
    model = _get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    vectors = model.encode(
        [str(chunk["text"]) for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError(f"Embedding model returned an invalid vector shape: {vectors.shape}")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks


def _answer_uploaded_question(
    question: str,
    index: Any,
    chunks: list[dict[str, Any]],
    top_k: int,
    min_score: float,
    min_margin: float,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Enter a question before asking.")
    model = _get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    if vector.shape != (1, index.d):
        raise ValueError(
            f"Question encoder returned shape {vector.shape}; expected (1, {index.d})."
        )

    count = min(max(top_k, 2), index.ntotal)
    scores, positions = index.search(vector, count)
    margin = (
        float(scores[0][0] - scores[0][1])
        if count > 1 and positions[0][1] >= 0
        else None
    )
    retrieved = []
    for score, position in zip(scores[0][:top_k], positions[0][:top_k]):
        if position < 0:
            continue
        chunk = chunks[int(position)]
        retrieved.append(
            {
                "citation_id": f"S{len(retrieved) + 1}",
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "score": round(float(score), 6),
                "text": chunk["text"],
            }
        )

    confident = bool(retrieved) and retrieved[0]["score"] >= min_score
    if margin is not None and margin < min_margin:
        confident = False
    if confident:
        best = retrieved[0]
        citation = f"[{best['citation_id']}]"
        if best["page"] is not None:
            citation = f"[{best['citation_id']} · page {best['page']}]"
        answer = f"{best['text']} {citation}"
    else:
        answer = "I don't know based on the uploaded documents."
    return {
        "question": question.strip(),
        "answer": answer,
        "answerable": confident,
        "score_margin": margin,
        "retrieved": retrieved,
    }


def _report_files() -> list[Path]:
    return sorted(
        DEFAULT_OUTPUT_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("metrics"), dict):
        raise ValueError(f"Not a RAG evaluation report: {path}")
    return report


def _format_rate(metric: dict[str, Any]) -> str:
    rate = metric.get("rate")
    if rate is None:
        return "n/a"
    low, high = metric["wilson_95_ci"]
    return f"{rate:.1%} ({metric['successes']}/{metric['total']})\n95% CI {low:.1%}–{high:.1%}"


def _show_report(report: dict[str, Any], report_path: Path | None) -> None:
    metrics = report["metrics"]
    config = report.get("configuration", {})
    dataset = report.get("dataset", {})

    st.caption(
        f"Run: {report.get('run_at_utc', 'unknown')} · "
        f"Model: {config.get('embedding_model', 'unknown')} · "
        f"Cases: {dataset.get('case_count', metrics.get('case_count', 'unknown'))}"
    )
    st.caption(
        f"top_k={config.get('top_k', 'n/a')} · "
        f"min_score={config.get('min_score', 'n/a')} · "
        f"min_margin={config.get('min_margin', 'n/a')}"
    )

    columns = st.columns(4)
    columns[0].metric("Source Hit@k", _format_rate(metrics["source_hit_at_k"]))
    columns[1].metric(
        "Answer behavior",
        _format_rate(metrics["answer_behavior_accuracy"]),
    )
    mrr = metrics.get("mean_reciprocal_rank_at_k")
    columns[2].metric("Mean MRR@k", f"{mrr:.3f}" if mrr is not None else "n/a")
    precision = metrics.get("mean_source_precision_at_k")
    columns[3].metric(
        "Mean Precision@k",
        f"{precision:.3f}" if precision is not None else "n/a",
    )

    more_columns = st.columns(4)
    recall = metrics.get("mean_source_recall_at_k")
    more_columns[0].metric(
        "Mean Recall@k", f"{recall:.3f}" if recall is not None else "n/a"
    )
    more_columns[1].metric(
        "No-answer accuracy", _format_rate(metrics["no_answer_accuracy"])
    )
    citation_rate = metrics.get("citation_id_validity_rate")
    more_columns[2].metric(
        "Valid citation IDs",
        _format_rate(citation_rate) if citation_rate else "n/a",
    )
    term_coverage = metrics.get("mean_expected_term_coverage")
    more_columns[3].metric(
        "Expected-term coverage",
        f"{term_coverage:.1%}" if term_coverage is not None else "n/a",
    )

    caveats = report.get("caveats", [])
    if caveats:
        with st.expander("Evaluation caveats", expanded=False):
            for caveat in caveats:
                st.warning(caveat)

    cases = report.get("cases", [])
    st.subheader("Case-by-case results")
    if not cases:
        st.info("This report contains no per-case results.")
    else:
        case_ids = [str(case.get("case_id", f"case-{index + 1}")) for index, case in enumerate(cases)]
        selected_id = st.selectbox("Select a test case", case_ids)
        selected = cases[case_ids.index(selected_id)]
        status = "PASS" if selected.get("answer_behavior_correct") else "FAIL"
        st.markdown(f"**{status}** · {'Answerable' if selected.get('expected_answerable') else 'Expected to abstain'}")
        st.markdown(f"**Question**  \n{selected.get('query', '')}")
        st.markdown(f"**System output**  \n{selected.get('answer', '')}")

        matched_terms = selected.get("matched_expected_terms", [])
        expected_terms = selected.get("expected_terms", [])
        if expected_terms:
            st.caption(
                f"Expected terms found: {', '.join(matched_terms) if matched_terms else 'none'} "
                f"(coverage {selected.get('expected_term_coverage', 0):.0%})"
            )

        retrieved = selected.get("retrieved", [])
        if retrieved:
            st.markdown("**Retrieved sources**")
            for item in retrieved:
                with st.expander(
                    f"{item.get('citation_id', '?')} · {item.get('source', 'unknown')} "
                    f"· score {item.get('score', 0):.3f}"
                ):
                    st.write(item.get("text", ""))
        else:
            st.info("No source passages were retrieved.")

    if report_path is not None:
        st.caption(f"Report file: `{report_path}`")
    st.download_button(
        "Download this report (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name=report_path.name if report_path else "rag-evaluation.json",
        mime="application/json",
    )


def _render_upload_tab() -> None:
    st.subheader("Ask questions about your documents")
    st.write(
        "Upload text, Markdown, source-code/data text files, or searchable PDFs. "
        "The app retrieves relevant passages and answers by quoting the top passage."
    )
    st.info(
        "Uploads are processed in memory for this Streamlit session and are not added "
        "to the demo corpus or written to disk. Scanned PDFs need OCR before upload."
    )
    st.caption(
        "Limits: 20 files, 10 MB per file, 25 MB total, 100 pages per PDF, "
        "and 500,000 extracted characters."
    )

    if "upload_widget_version" not in st.session_state:
        st.session_state["upload_widget_version"] = 0
    uploader_key = f"document_uploader_{st.session_state['upload_widget_version']}"
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        key=uploader_key,
        help="Supported: PDF, TXT, Markdown, Pine Script, Python, JavaScript/TypeScript, JSON/JSONL, CSV, LOG, YAML.",
    )

    fingerprint = _uploads_fingerprint(uploaded_files) if uploaded_files else None
    existing_index = st.session_state.get("uploaded_rag_index")
    indexed_fingerprint = st.session_state.get("uploaded_rag_fingerprint")
    upload_col, clear_col = st.columns([3, 1])
    with upload_col:
        index_clicked = st.button(
            "Index uploaded documents",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )
    with clear_col:
        clear_clicked = st.button("Clear session", use_container_width=True)

    if clear_clicked:
        for key in (
            "uploaded_rag_index",
            "uploaded_rag_chunks",
            "uploaded_rag_fingerprint",
            "uploaded_rag_files",
            "uploaded_rag_history",
        ):
            st.session_state.pop(key, None)
        st.session_state["upload_widget_version"] += 1
        st.rerun()

    if index_clicked:
        try:
            with st.spinner("Reading documents and building an in-memory search index..."):
                index, chunks = _build_upload_index(uploaded_files)
            st.session_state["uploaded_rag_index"] = index
            st.session_state["uploaded_rag_chunks"] = chunks
            st.session_state["uploaded_rag_fingerprint"] = fingerprint
            st.session_state["uploaded_rag_files"] = sorted(
                {str(chunk["source"]) for chunk in chunks}
            )
            st.session_state["uploaded_rag_history"] = []
            st.success(
                f"Indexed {len(uploaded_files)} file(s) into {len(chunks)} searchable passages."
            )
            existing_index = index
            indexed_fingerprint = fingerprint
        except ValueError as error:
            st.error(str(error))

    if existing_index is not None:
        if fingerprint and fingerprint != indexed_fingerprint:
            st.warning("The selected files differ from the indexed files. Re-index before asking.")
        else:
            indexed_files = st.session_state.get("uploaded_rag_files", [])
            st.caption(
                f"Indexed {len(indexed_files)} file(s): {', '.join(indexed_files)}"
            )
            top_k_col, score_col, margin_col = st.columns(3)
            top_k = int(top_k_col.number_input(
                "Passages to retrieve",
                min_value=1,
                max_value=10,
                value=3,
                key="upload_top_k",
            ))
            min_score = score_col.number_input(
                "Minimum similarity score",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_MIN_SCORE,
                step=0.01,
                format="%.2f",
                key="upload_min_score",
            )
            min_margin = margin_col.number_input(
                "Minimum top-score margin",
                min_value=0.0,
                max_value=2.0,
                value=DEFAULT_MIN_MARGIN,
                step=0.01,
                format="%.2f",
                key="upload_min_margin",
            )

            with st.form("uploaded_document_question", clear_on_submit=True):
                question = st.text_input(
                    "Ask a question about the uploaded files",
                    placeholder="What does the document say about ...?",
                )
                ask_clicked = st.form_submit_button("Ask", type="primary")
            if ask_clicked:
                if not question.strip():
                    st.error("Enter a question before asking.")
                else:
                    with st.spinner("Searching the uploaded documents..."):
                        response = _answer_uploaded_question(
                            question,
                            st.session_state["uploaded_rag_index"],
                            st.session_state["uploaded_rag_chunks"],
                            top_k,
                            float(min_score),
                            float(min_margin),
                        )
                    st.session_state.setdefault("uploaded_rag_history", []).append(response)

            history = st.session_state.get("uploaded_rag_history", [])
            for response in reversed(history):
                with st.chat_message("user"):
                    st.text(response["question"])
                with st.chat_message("assistant"):
                    st.text(response["answer"])
                    if response["retrieved"]:
                        with st.expander("Retrieved passages and scores"):
                            for passage in response["retrieved"]:
                                page_label = (
                                    f", page {passage['page']}"
                                    if passage["page"] is not None else ""
                                )
                                st.markdown(
                                    f"**[{passage['citation_id']}] "
                                    f"{passage['source']}{page_label}** · "
                                    f"score {passage['score']:.3f}"
                                )
                                st.text(passage["text"])
                    else:
                        st.caption("No passages were found.")


def _render_evaluation_tab() -> None:
    settings_col, run_col = st.columns([2, 1])
    with settings_col:
        top_k = st.number_input(
            "Top-k passages",
            min_value=1,
            max_value=50,
            value=3,
            key="eval_top_k",
        )
        threshold_col, margin_col = st.columns(2)
        min_score = threshold_col.number_input(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_MIN_SCORE,
            step=0.01,
            format="%.2f",
            key="eval_min_score",
        )
        min_margin = margin_col.number_input(
            "Minimum top-score margin",
            min_value=0.0,
            max_value=2.0,
            value=DEFAULT_MIN_MARGIN,
            step=0.01,
            format="%.2f",
            key="eval_min_margin",
        )
    with run_col:
        st.write("")
        st.write("")
        run_clicked = st.button("Run evaluation", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Loading the embedding model and evaluating cases..."):
            report = evaluate(
                dataset_path=DEFAULT_DATASET,
                store_dir=DEFAULT_STORE,
                top_k=int(top_k),
                min_score=float(min_score),
                min_margin=float(min_margin),
            )
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            report_path = DEFAULT_OUTPUT_DIR / f"evaluation-{stamp}.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        st.session_state["rag_eval_report"] = report
        st.session_state["rag_eval_report_path"] = str(report_path)
        st.success("Evaluation completed and report saved.")

    report = st.session_state.get("rag_eval_report")
    saved_path_text = st.session_state.get("rag_eval_report_path")
    report_path = Path(saved_path_text) if saved_path_text else None

    if report is None:
        files = _report_files()
        if files:
            selected_report_text = st.selectbox(
                "Saved evaluation report",
                [str(path) for path in files],
                format_func=lambda path: Path(path).name,
            )
            selected_report = Path(selected_report_text)
            try:
                report = _read_report(selected_report)
                report_path = selected_report
            except (OSError, json.JSONDecodeError, ValueError) as error:
                st.error(f"Could not load the selected report: {error}")
        else:
            st.info("No saved report yet. Choose settings and click **Run evaluation**.")

    if report is not None:
        _show_report(report, report_path)

    st.caption(
        "The included demo dataset is small and synthetic. Its scores are an illustration, "
        "not independent evidence of production quality."
    )


def main() -> None:
    st.set_page_config(page_title="RAG Evaluation Suite", page_icon="🔎", layout="wide")
    st.title("RAG Evaluation Suite")
    st.write(
        "Explore your own documents, or evaluate the included retrieval benchmark."
    )
    upload_tab, evaluation_tab = st.tabs(["Upload & ask", "Evaluation benchmark"])
    with upload_tab:
        _render_upload_tab()
    with evaluation_tab:
        _render_evaluation_tab()


if __name__ == "__main__":
    main()
