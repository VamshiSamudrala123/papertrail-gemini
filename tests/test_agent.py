"""Exercise provider routing and retrieval without API requests."""
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from agentic_rag import DEFAULT_GEMINI_MODEL, Evidence, ask_agent

class AgentTests(unittest.TestCase):
    def setUp(self):
        self.sdk = ModuleType("crewai")
        for name in ("LLM", "Agent", "Task", "Crew"):
            setattr(self.sdk, name, Mock())
        tool_module = ModuleType("crewai.tools")
        tool_module.tool = lambda name: lambda function: function
        modules = patch.dict(sys.modules, {"crewai": self.sdk, "crewai.tools": tool_module})
        modules.start()
        self.addCleanup(modules.stop)
        self.index = Mock()
        self.index.search.return_value = [Evidence("r1", "report.pdf", 2, "Revenue was 42 million.", 0.1)]
        self.sdk.Crew.return_value.kickoff.side_effect = self.run_search

    def run_search(self):
        search = self.sdk.Agent.call_args.kwargs["tools"][0]
        self.excerpts = search("annual revenue")
        search("revenue details")
        return SimpleNamespace(raw="Revenue was 42 million. [S1]")

    def test_gemini_search_returns_evidence_and_history(self):
        answer, evidence = ask_agent(self.index, "And revenue?", [
            {"role": "user", "content": "Tell me about the report"},
        ], "Gemini", DEFAULT_GEMINI_MODEL, " test-google-key ", top_k=3)
        self.sdk.LLM.assert_called_once_with(
            model="gemini/gemini-3.8-flash", api_key="test-google-key", timeout=90,
        )
        self.assertEqual(evidence, self.index.search.return_value)
        self.assertIn("[S1] report.pdf, page 2", self.excerpts)
        self.assertIn("[S1]", answer)
        self.index.search.assert_any_call("annual revenue", k=3)
        description = self.sdk.Task.call_args.kwargs["description"]
        self.assertIn("Tell me about the report", description)
        self.assertNotIn("test-google-key", description)

    def test_blank_or_prefixed_model_is_normalized(self):
        for model in ("", "  ", "gemini/gemini-3.8-flash"):
            with self.subTest(model=model):
                self.sdk.LLM.reset_mock()
                ask_agent(self.index, "Revenue?", [], "Gemini", model, "key")
                self.assertEqual(self.sdk.LLM.call_args.kwargs["model"], "gemini/gemini-3.8-flash")

    def test_missing_key_fails_before_constructing_provider(self):
        for key in (None, "", "  "):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Google API key"):
                ask_agent(self.index, "Revenue?", [], "Gemini", DEFAULT_GEMINI_MODEL, key)
        self.sdk.LLM.assert_not_called()

    def test_no_evidence_does_not_return_unverified_answer(self):
        self.index.search.return_value = []
        answer, evidence = ask_agent(self.index, "Revenue?", [], "Gemini", DEFAULT_GEMINI_MODEL, "key")
        self.assertEqual(evidence, [])
        self.assertIn("couldn't verify", answer)
        self.assertNotIn("42 million", answer)

    def test_ollama_does_not_require_google_key(self):
        ask_agent(self.index, "Revenue?", [], "Ollama", "llama3.2")
        self.assertEqual(self.sdk.LLM.call_args.kwargs["model"], "ollama/llama3.2")
        self.assertNotIn("api_key", self.sdk.LLM.call_args.kwargs)

    def test_unknown_provider_does_not_fall_back(self):
        with self.assertRaisesRegex(ValueError, "Unknown model provider"):
            ask_agent(self.index, "Revenue?", [], "Other", "model", "key")
        self.sdk.LLM.assert_not_called()
