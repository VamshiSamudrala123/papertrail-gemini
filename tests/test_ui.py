"""UI smoke tests use Streamlit's runner and a small in-memory document index."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentic_rag import Evidence, Page

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "Install requirements-dev.txt for Streamlit UI tests")
class WorkspaceTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.secrets["GOOGLE_API_KEY"] = "test-key"
        return app

    def button(self, app, label):
        return next(button for button in app.button if button.label == label)

    def test_empty_workspace_loads_and_requires_documents(self):
        app = self.app().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertEqual([tab.label for tab in app.tabs], ["Conversation", "Search quality", "Guide"])
        self.assertTrue(any("Follow the evidence." in block.value for block in app.markdown))

    def test_starter_question_renders_citations_and_new_chat_keeps_documents(self):
        upload = SimpleNamespace(name="report.pdf", getvalue=lambda: b"test-pdf")
        index = FakeIndex()
        with patch("streamlit.file_uploader", return_value=[upload]), patch(
            "agentic_rag.read_pdfs", return_value=([Page("report.pdf", 2, "Revenue was 42 million.")], [])
        ), patch("agentic_rag.DocumentIndex", return_value=index), patch(
            "agentic_rag.ask_agent", return_value=("Revenue was 42 million. [S1]", index.search(""))
        ) as ask:
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.chat_input[0].disabled)
            self.button(app, "Give me the big picture").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["messages"]), 2)
            self.assertIn("Summarize the main points", ask.call_args.args[1])
            self.assertEqual(app.chat_message[-1].markdown[0].value, "Revenue was 42 million. [S1]")
            self.assertTrue(any('class="source-id">S1' in block.value for block in app.markdown))
            self.button(app, "New conversation").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["messages"], [])
            self.assertIs(app.session_state["index"], index)

    def test_evaluation_survives_widget_rerun_and_clear_resets_workspace(self):
        upload = SimpleNamespace(name="report.pdf", getvalue=lambda: b"test-pdf")
        index = FakeIndex()
        with patch("streamlit.file_uploader", return_value=[upload]) as uploader, patch(
            "agentic_rag.read_pdfs", return_value=([Page("report.pdf", 2, "Revenue was 42 million.")], [])
        ), patch("agentic_rag.DocumentIndex", return_value=index):
            app = self.app().run()
            app.text_area[0].set_value('{"question":"Revenue?","expected_source":"report.pdf"}').run()
            self.button(app, "Run evaluation").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["evaluation"]["metrics"]["hit_at_k"], 1.0)
            app.slider[0].set_value(4).run()
            self.assertEqual(app.session_state["evaluation"]["k"], 5)
            uploader.return_value = []
            self.button(app, "Clear workspace").click().run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["index"])
            self.assertIsNone(app.session_state["evaluation"])
            self.assertTrue(index.closed)


class FakeIndex:
    closed = False

    def add(self, chunks):
        pass

    def search(self, question, k=5):
        return [Evidence("1", "report.pdf", 2, "Revenue was 42 million.", 0.1)]

    def close(self):
        self.closed = True
