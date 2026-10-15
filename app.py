"""Streamlit workspace for reading, asking, and checking document evidence."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path

import streamlit as st

from request_errors import AnswerUnavailable, answer_with_retry, classify_error

from agentic_rag import (
    DEFAULT_GEMINI_MODEL, DocumentIndex, ask_agent, evaluate_retrieval,
    read_pdfs, split_pages,
)

st.set_page_config(
    page_title="Papertrail · Read between the pages", page_icon="📄",
    layout="wide", initial_sidebar_state="expanded",
)
st.markdown(f"<style>{Path(__file__).with_name('styles.css').read_text()}</style>", unsafe_allow_html=True)

DEFAULTS = {
    "messages": [], "index": None, "fingerprint": None,
    "sources": [], "chunks": 0, "evaluation": None, "pending_question": None, "failed_question": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0


def clear_documents():
    if st.session_state.index is not None:
        st.session_state.index.close()
    for key, value in DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value


def reset_workspace():
    clear_documents()
    st.session_state.upload_version += 1


def new_chat():
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.failed_question = None


def queue_question(question):
    st.session_state.pending_question = question


def configured_google_key():
    environment_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if environment_key:
        return environment_key
    try:
        return st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    except FileNotFoundError:
        return ""


def show_evidence(evidence):
    for number, source in enumerate(evidence, 1):
        # PDF text and filenames are data, so escape them before rendering HTML.
        name = html.escape(str(source["source"]))
        page = html.escape(str(source["page"]))
        passage = html.escape(str(source["text"]))
        st.markdown(
            f'<article class="evidence-card"><div class="evidence-heading">'
            f'<span class="source-id">S{number}</span><span class="source-name">{name}</span>'
            f'<span class="page-number">p. {page}</span></div>'
            f'<p class="passage">{passage}</p></article>',
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">▤</span> papertrail<span class="brand-dot">.</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-tagline">A little clarity in every document.</p>', unsafe_allow_html=True)
    st.button("New conversation", on_click=new_chat, use_container_width=True)
    st.markdown('<div class="section-label">YOUR DOCUMENTS</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Add PDFs", type="pdf", accept_multiple_files=True,
        key=f"uploads_{st.session_state.upload_version}",
        help="Up to 8 PDFs, 12 MB each. Use PDFs with selectable text.",
        label_visibility="collapsed",
    )

if uploads:
    raw = [(file.name, file.getvalue()) for file in uploads]
    fingerprint = hashlib.sha256(b"".join(name.encode() + hashlib.sha256(data).digest() for name, data in raw)).hexdigest()
    if fingerprint != st.session_state.fingerprint:
        try:
            with st.spinner("Getting your documents ready…"):
                pages, warnings = read_pdfs(raw)
                chunks = split_pages(pages)
                fresh = DocumentIndex()
                try:
                    fresh.add(chunks)
                except Exception:
                    fresh.close()
                    raise
                clear_documents()
                st.session_state.index = fresh
                st.session_state.fingerprint = fingerprint
                st.session_state.sources = sorted({page.source for page in pages})
                st.session_state.chunks = len(chunks)
            for warning in warnings:
                st.warning(warning)
        except Exception as exc:
            clear_documents()
            st.error(f"Could not prepare these documents: {exc}")
elif st.session_state.index is not None:
    clear_documents()

index = st.session_state.index
with st.sidebar:
    if index is not None:
        count = len(st.session_state.sources)
        st.caption(f"{count} document{'s' if count != 1 else ''} · {st.session_state.chunks} searchable passages")
    else:
        st.caption("Add a report, paper, or anything you want to understand. Up to 8 PDFs · 12 MB each.")

    st.markdown('<div class="section-label">CONNECTION</div>', unsafe_allow_html=True)
    saved_key = configured_google_key()
    with st.expander("Google API key", expanded=not bool(saved_key or st.session_state.get("google_api_key"))):
        entered_key = st.text_input(
            "Your Google API key", type="password", key="google_api_key",
            placeholder="Paste your key here",
            help="You can also configure GOOGLE_API_KEY or GEMINI_API_KEY in Streamlit secrets.",
        )
        st.caption("Questions, recent chat, and retrieved passages go to Google and use your API quota.")
        st.markdown("[Get a key from Google AI Studio](https://aistudio.google.com/apikey)")
    api_key = entered_key.strip() or saved_key
    st.markdown(
        '<div class="connection-status">' + ("● Google key added" if api_key else "○ Google key needed") + '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Model & search settings"):
        provider = st.selectbox("Answer mode", ["Gemini", "Ollama", "Search only"])
        model = ""
        if provider == "Gemini":
            model = st.text_input("Gemini model", value=DEFAULT_GEMINI_MODEL, key="gemini_model")
        elif provider == "Ollama":
            model = st.text_input("Ollama model", value="llama3.2", key="ollama_model")
        k = st.slider("Passages per search", 2, 10, 5)
    st.button("Clear workspace", on_click=reset_workspace, use_container_width=True)
    st.caption("Documents and conversations last for this session.")

ready = index is not None and (provider != "Gemini" or bool(api_key))
mode_label = "Gemini" if provider == "Gemini" else provider
st.markdown(
    '<div class="workspace-header"><span class="section-label">DOCUMENT WORKSPACE</span>'
    f'<span class="mode-pill"><span>●</span> {html.escape(mode_label)}</span></div>',
    unsafe_allow_html=True,
)

chat, evaluation_tab, guide = st.tabs(["Conversation", "Search quality", "Guide"])
with chat:
    conversation, sources = st.columns([2.2, 1], gap="large")
    with conversation:
        heading, actions = st.columns([3, 1])
        with heading:
            st.markdown('<h2 class="conversation-title">The reading room</h2>', unsafe_allow_html=True)
        with actions:
            if st.session_state.messages:
                st.download_button(
                    "Export chat", json.dumps(st.session_state.messages, indent=2),
                    "papertrail-chat.json", "application/json", use_container_width=True,
                )
        thread = st.container(key="conversation")
        with thread:
            if not st.session_state.messages:
                st.markdown(
                    '<div class="welcome"><div class="welcome-symbol">▤</div>'
                    '<p class="eyebrow">LESS SEARCHING. MORE UNDERSTANDING.</p>'
                    '<h1>Follow the evidence.</h1>'
                    '<p>Ask a question. Connect the details.<br>Trace every answer back to the page.</p></div>',
                    unsafe_allow_html=True,
                )
                if index is None:
                    st.markdown('<p class="next-step">Start by adding a PDF in the left panel.</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="next-step">Your documents are ready. Start with a question below.</p>', unsafe_allow_html=True)
                    for title, prompt in [
                        ("Give me the big picture", "Summarize the main points of the uploaded documents, with source citations."),
                        ("Pull out the key facts", "What are the most important facts and figures in these documents? Cite the supporting pages."),
                        ("What should I look at more closely?", "Which claims, caveats, or unresolved questions in these documents deserve closer attention? Use only document evidence and cite the pages."),
                    ]:
                        st.button(title, on_click=queue_question, args=(prompt,), disabled=not ready, use_container_width=True)
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar="📄" if message["role"] == "assistant" else None):
                    st.markdown(message["content"])
                    if message.get("evidence"):
                        with st.expander(f"Check {len(message['evidence'])} source passages"):
                            show_evidence(message["evidence"])

        if index is not None and provider == "Gemini" and not api_key:
            st.info("Add your Google API key in the left panel to start asking questions.")
        failed = st.session_state.failed_question
        if failed:
            with thread:
                st.warning(failed["message"])
                st.button(
                    "Try again", on_click=queue_question, args=(failed["question"],),
                    disabled=not ready, key="retry_answer",
                )
        typed_question = st.chat_input("What would you like to know?", disabled=not ready)
        question = typed_question or st.session_state.pending_question
        st.session_state.pending_question = None
        if question and ready:
            retrying = bool(failed and failed["question"] == question)
            history = list(st.session_state.messages)
            if retrying and history and history[-1].get("role") == "user" and history[-1].get("content") == question:
                history = history[:-1]
            else:
                st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.failed_question = None
            with thread:
                if not retrying:
                    with st.chat_message("user"):
                        st.markdown(question)
                progress = st.empty()
                try:
                    with st.spinner("Reading the relevant pages…"):
                        if provider == "Search only":
                            found = index.search(question, k)
                            answer = "Here are the closest passages from your documents." if found else "No passages found."
                        else:
                            answer, found = answer_with_retry(
                                lambda: ask_agent(index, question, history, provider, model, api_key, k),
                                on_retry=lambda attempt: progress.info(f"The model is temporarily busy. Trying again ({attempt} of 3)…"),
                            )
                    evidence = [{"source": e.source, "page": e.page, "text": e.text} for e in found]
                    st.session_state.messages.append({"role": "assistant", "content": answer, "evidence": evidence})
                except Exception as exc:
                    issue = exc.issue if isinstance(exc, AnswerUnavailable) else classify_error(exc)
                    st.session_state.failed_question = {"question": question, "message": issue.message}
                finally:
                    progress.empty()
            # Rerender once so the welcome screen disappears and saved errors remain visible.
            st.rerun()
        st.caption("Answers can miss context. Check the source passages before relying on a claim.")

    with sources:
        with st.container(key="evidence"):
            st.markdown('<div class="section-label">THE PAPER TRAIL</div><h3 class="evidence-title">Behind the answer</h3>', unsafe_allow_html=True)
            latest = next((message for message in reversed(st.session_state.messages) if message.get("role") == "assistant"), None)
            if latest and latest.get("evidence"):
                st.caption("Passages retrieved for the latest answer. Match S1, S2, … to its citations.")
                show_evidence(latest["evidence"])
            else:
                st.markdown(
                    '<div class="sources-empty"><div class="paper-stack">▤</div>'
                    '<h4>A source for every detail.</h4>'
                    '<p>After you ask a question, the passages used for the answer appear here, with document names and page numbers.</p></div>',
                    unsafe_allow_html=True,
                )
                if st.session_state.sources:
                    st.markdown('<div class="section-label">ON YOUR DESK</div>', unsafe_allow_html=True)
                    for source in st.session_state.sources:
                        st.markdown(f'<div class="desk-document">▤ <span>{html.escape(source)}</span></div>', unsafe_allow_html=True)

with evaluation_tab:
    st.subheader("How well does search find the right page?")
    st.write("Check retrieval against questions with known sources. These scores measure passage search, not answer correctness.")
    with st.expander("Dataset format", expanded=True):
        st.caption("One JSON object per line. Include an expected filename, a phrase, or both.")
        st.code('{"question":"What was revenue?","expected_source":"report.pdf","expected_text":"revenue"}', language="json")
    pasted = st.text_area("Your evaluation questions", height=160, placeholder="Paste your JSONL dataset…")
    if st.button("Run evaluation", disabled=index is None, type="primary"):
        try:
            cases = [json.loads(line) for line in pasted.splitlines() if line.strip()]
            metrics, rows = evaluate_retrieval(index, cases, k)
            st.session_state.evaluation = {"metrics": metrics, "rows": rows, "k": k}
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
    if st.session_state.evaluation:
        result = st.session_state.evaluation
        metrics = result["metrics"]
        a, b, c = st.columns(3)
        a.metric("Questions", metrics["questions"])
        b.metric(f"Hit@{result['k']}", f"{metrics['hit_at_k']:.1%}")
        c.metric(f"MRR@{result['k']}", f"{metrics['mrr_at_k']:.3f}")
        st.dataframe(result["rows"], use_container_width=True)
        st.download_button("Export evaluation", json.dumps(result, indent=2), "papertrail-evaluation.json", "application/json")

with guide:
    st.subheader("From a question to a page")
    st.markdown("""
1. **Add your documents.** Upload PDFs with selectable text. Scanned pages need OCR first.
2. **Ask naturally.** Start with a summary or a specific detail. Follow-up questions use recent chat context.
3. **Check the evidence.** Read the numbered passages beside the answer and open earlier sources within the conversation.

Use **Search quality** to test whether retrieval finds pages you already know are relevant. Export a conversation before closing the session if you want to keep it.

**Under the hood:** page-aware chunks, local MiniLM embeddings, Chroma search, and a CrewAI agent that can search repeatedly. Gemini receives the question, recent chat, and retrieved passages. Source passages are real search results, but the model can still misinterpret them or cite them incorrectly.

Ollama needs a server on the same machine as this app. Search only mode needs no API key.
""")
