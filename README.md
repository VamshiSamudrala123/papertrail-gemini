# Papertrail

Ask questions across PDFs and check the passages behind each answer. A CrewAI agent searches a session-scoped Chroma index and uses **Gemini 3.8 Flash** (`gemini-3.8-flash`) to explain what it finds.

Inspired by [Aman Kharwal's Agentic RAG walkthrough](https://amanxai.com/2026/07/08/build-an-end-to-end-agentic-rag-system/), with a Streamlit interface, multiple documents, page citations, chat export, and retrieval evaluation.

## Run locally

Use Python 3.11 or 3.12.

```bash
git clone https://github.com/VamshiSamudrala123/papertrail-gemini.git
cd papertrail-gemini
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`.

Upload up to eight text-based PDFs, 12 MB each. The first run downloads `sentence-transformers/all-MiniLM-L6-v2`; indexing runs locally on the app server. Scanned PDFs need OCR first.

## Connect Gemini

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey). Enter it in the sidebar's password field, set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in your environment, or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add it there.

```toml
GOOGLE_API_KEY = "your-key"
```

The sidebar key takes priority. Otherwise environment variables take priority over Streamlit secrets; `GOOGLE_API_KEY` is checked before `GEMINI_API_KEY` within each source. Keys stay out of chat exports and should never be committed.

Gemini receives the question, recent conversation, and passages retrieved by the agent. Embeddings remain local. Requests use your Google account's quota and billing. The model field is editable if you need another available Gemini model; there is no silent fallback to another provider. See [Google's model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) for availability and [CrewAI's provider setup](https://docs.crewai.com/en/concepts/llms) for the integration.

For local answers with Ollama, run `ollama pull llama3.2` and start Ollama on `localhost:11434`. Search only mode needs no language model or API key.

## What's included

- Page-aware PDF chunks, MiniLM embeddings, and Chroma semantic search.
- A CrewAI agent that can search repeatedly before answering.
- Citation markers with expandable filenames, page numbers, and source passages.
- Session chat history and JSON export.
- A retrieval evaluation tab with Hit@k, MRR@k, and result export.

## Evaluate retrieval

Paste one JSON object per line into the evaluation tab:

```jsonl
{"question":"What was revenue?","expected_source":"annual.pdf","expected_text":"revenue"}
{"question":"Which risks are mentioned?","expected_source":"annual.pdf"}
```

Each question needs at least one expected label. A hit requires a single passage to match every provided label. Hit@k measures the fraction of questions with a hit among the first k passages; MRR@k averages the reciprocal rank of the first hit. These measure retrieval, not answer quality.

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The tests use doubles for the agent and index and do not spend API credits. For a live check, upload a short PDF, ask a question with a known answer, inspect its page citation, then ask something the PDF does not cover.

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select `VamshiSamudrala123/papertrail-gemini`, branch `main`, entrypoint `app.py`.
3. Choose Python 3.11 or 3.12.
4. Add `GOOGLE_API_KEY = "your-key"` under the app's Secrets settings, or enter your key in the sidebar at runtime.
5. Deploy, upload a small PDF, and check a cited answer.

A shared server key lets visitors use its quota. For a public demo, leave it unset and have visitors enter their own keys. Ollama requires a server on the same machine, so use Gemini or Search only on Community Cloud. The embedding model and dependencies can exceed small hosting instances' memory limits.

## Limits

Collections and chat history are temporary and disappear when the session ends. The agent may misread a passage or attach an inaccurate citation; always check the excerpts. Source passages shown in the interface come from actual searches, but their presence does not prove every answer claim. PDF content is treated as untrusted data. This demo has no authentication, persistent storage, OCR, or grounded-answer scoring.
