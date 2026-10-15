from __future__ import annotations

import unittest

import pymupdf

from rag_eval.uploads import chunk_uploaded_files


class UploadTests(unittest.TestCase):
    def test_chunks_utf8_text_and_preserves_source(self) -> None:
        chunks = chunk_uploaded_files(
            [("notes.md", ("alpha " * 500).encode("utf-8"))]
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["source"] == "notes.md" for chunk in chunks))
        self.assertTrue(all(chunk["page"] is None for chunk in chunks))
        self.assertTrue(all(len(str(chunk["text"])) <= 1200 for chunk in chunks))

    def test_extracts_pdf_text_and_page_number(self) -> None:
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Hello from uploaded PDF.")
        content = document.tobytes()
        document.close()

        chunks = chunk_uploaded_files([("guide.pdf", content)])
        self.assertEqual(chunks[0]["source"], "guide.pdf")
        self.assertEqual(chunks[0]["page"], 1)
        self.assertIn("Hello from uploaded PDF.", str(chunks[0]["text"]))

    def test_rejects_unsupported_file_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported file type"):
            chunk_uploaded_files([("archive.exe", b"not executable")])

    def test_rejects_invalid_utf8(self) -> None:
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            chunk_uploaded_files([("notes.txt", b"\xff\xfe")])

    def test_rejects_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate uploaded filename"):
            chunk_uploaded_files([("same.txt", b"first"), ("same.txt", b"second")])

    def test_rejects_oversized_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "10 MB per-file limit"):
            chunk_uploaded_files([("large.txt", b"x" * (10 * 1024 * 1024 + 1))])


if __name__ == "__main__":
    unittest.main()
