"""Validate and chunk session-only user document uploads."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable

import pymupdf

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
MAX_TOTAL_CHARACTERS = 500_000
MAX_FILES = 20
MAX_PDF_PAGES = 100
CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 180

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pine",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".jsonl",
    ".csv",
    ".log",
    ".yaml",
    ".yml",
}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"}


def _display_name(filename: str) -> str:
    name = PurePosixPath(filename.replace("\\", "/")).name
    if not name or name in {".", ".."} or "\x00" in name:
        raise ValueError("Uploaded file has an invalid filename.")
    return name


def _extract_pdf(filename: str, content: bytes) -> list[dict[str, object]]:
    try:
        pdf = pymupdf.open(stream=content, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise ValueError(f"{filename}: could not read PDF: {error}") from error

    with pdf:
        if pdf.needs_pass:
            raise ValueError(f"{filename}: encrypted PDFs are not supported.")
        if len(pdf) > MAX_PDF_PAGES:
            raise ValueError(
                f"{filename}: has {len(pdf)} pages; the limit is {MAX_PDF_PAGES}."
            )
        pages = [
            {"source": filename, "page": page_number, "text": page.get_text("text").strip()}
            for page_number, page in enumerate(pdf, start=1)
        ]
    if not any(page["text"] for page in pages):
        raise ValueError(f"{filename}: no selectable text found; scanned PDFs need OCR first.")
    return pages


def _extract_text(filename: str, content: bytes) -> list[dict[str, object]]:
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise ValueError(f"{filename}: text files must use UTF-8 encoding.") from error
    if not text:
        raise ValueError(f"{filename}: file is empty.")
    return [{"source": filename, "page": None, "text": text}]


def chunk_uploaded_files(files: Iterable[tuple[str, bytes]]) -> list[dict[str, object]]:
    """Extract and chunk uploads without writing their contents to disk."""
    uploaded = list(files)
    if not uploaded:
        raise ValueError("Choose at least one file to upload.")
    if len(uploaded) > MAX_FILES:
        raise ValueError(f"Upload at most {MAX_FILES} files at a time.")

    total_bytes = sum(len(content) for _, content in uploaded)
    if total_bytes > MAX_TOTAL_BYTES:
        raise ValueError(
            f"Combined upload size exceeds {MAX_TOTAL_BYTES // (1024 * 1024)} MB."
        )

    names: set[str] = set()
    pages: list[dict[str, object]] = []
    for original_name, content in uploaded:
        filename = _display_name(original_name)
        if filename in names:
            raise ValueError(f"Duplicate uploaded filename: {filename}")
        names.add(filename)
        if len(content) > MAX_FILE_BYTES:
            raise ValueError(f"{filename}: exceeds the 10 MB per-file limit.")
        suffix = PurePosixPath(filename).suffix.casefold()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"{filename}: unsupported file type. Supported: {supported}")
        if suffix == ".pdf":
            pages.extend(_extract_pdf(filename, content))
        else:
            pages.extend(_extract_text(filename, content))

    total_characters = sum(len(str(page["text"])) for page in pages)
    if total_characters > MAX_TOTAL_CHARACTERS:
        raise ValueError(
            f"Extracted text exceeds the {MAX_TOTAL_CHARACTERS:,}-character limit."
        )

    chunks: list[dict[str, object]] = []
    for page in pages:
        text = " ".join(str(page["text"]).split())
        cursor = 0
        chunk_index = 0
        while cursor < len(text):
            end = min(cursor + CHUNK_SIZE, len(text))
            if end < len(text):
                boundary = text.rfind(" ", cursor + CHUNK_SIZE // 2, end)
                if boundary > cursor:
                    end = boundary
            chunk = text[cursor:end].strip()
            if chunk:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "chunk_index": chunk_index,
                        "source": page["source"],
                        "page": page["page"],
                        "text": chunk,
                    }
                )
                chunk_index += 1
            if end >= len(text):
                break
            next_cursor = max(end - CHUNK_OVERLAP, cursor + 1)
            boundary = text.find(" ", next_cursor, end)
            cursor = boundary + 1 if boundary >= 0 else next_cursor
    if not chunks:
        raise ValueError("No searchable text could be extracted from the uploaded files.")
    return chunks
