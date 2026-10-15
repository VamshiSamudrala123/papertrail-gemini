import io
import unittest
from pypdf import PdfWriter
from agentic_rag import Page, Evidence, read_pdfs, split_pages, evaluate_retrieval

class CoreTests(unittest.TestCase):
    def test_reject_non_pdf(self):
        with self.assertRaisesRegex(ValueError, "only PDF"):
            read_pdfs([("notes.txt", b"not pdf")])

    def test_empty_pdf(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)
        with self.assertRaisesRegex(ValueError, "readable text"):
            read_pdfs([("blank.pdf", buffer.getvalue())])

    def test_chunk_metadata(self):
        chunks = split_pages([Page("report.pdf", 3, "finance " * 200)], 180, 40)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(c.page == 3 and c.source == "report.pdf" for c in chunks))
        self.assertEqual(len(set(c.id for c in chunks)), len(chunks))

    def test_retrieval_metrics(self):
        metrics, rows = evaluate_retrieval(FakeIndex(), [
            {"question": "Revenue?", "expected_text": "revenue"},
            {"question": "Risk?", "expected_source": "else.pdf"},
        ])
        self.assertEqual(metrics["hit_at_k"], 0.5)
        self.assertEqual(metrics["mrr_at_k"], 0.5)
        self.assertEqual(rows[0]["rank"], 1)

    def test_missing_labels(self):
        with self.assertRaisesRegex(ValueError, "expected label"):
            evaluate_retrieval(FakeIndex(), [{"question": "Question"}])

class FakeIndex:
    def search(self, question, k=5):
        return [Evidence("id", "report.pdf", 2, "revenue was 42 million", 0.1)]
