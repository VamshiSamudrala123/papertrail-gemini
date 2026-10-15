import hashlib
import json
import os
import streamlit as st
from agentic_rag import DEFAULT_GEMINI_MODEL, DocumentIndex, ask_agent, evaluate_retrieval, read_pdfs, split_pages

st.set_page_config(page_title="Papertrail | PDF research", page_icon="📚", layout="wide")
st.title("📚 Papertrail")
st.caption("Your PDFs, searchable evidence, and a research agent.")
for key, default in {"messages": [], "index": None, "fingerprint": None, "sources": [], "chunks": 0}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def clear():
    if st.session_state.index is not None:
        st.session_state.index.close()
    for key, default in {"messages": [], "index": None, "fingerprint": None, "sources": [], "chunks": 0}.items():
        st.session_state[key] = default

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, help="Up to 8 files, 12 MB each")
    provider = st.selectbox("Answer mode", ["Gemini", "Ollama", "Search only"])
    model = ""
    api_key = None
    if provider == "Gemini":
        model = st.text_input("Model", value=DEFAULT_GEMINI_MODEL, key="gemini_model")
        try:
            secret_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
        except FileNotFoundError:
            secret_key = ""
        configured_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or secret_key
        entered_key = st.text_input("Google API key", type="password", key="google_api_key", help="Optional if GOOGLE_API_KEY or GEMINI_API_KEY is already configured.")
        api_key = entered_key.strip() or configured_key
        if api_key:
            st.caption("Google API key configured.")
        else:
            st.caption("Add a key from Google AI Studio to get Gemini answers.")
        st.caption("Your question and retrieved passages are sent to Google. API usage uses your Google quota.")
    elif provider == "Ollama":
        model = st.text_input("Model", value="llama3.2", key="ollama_model")
    k = st.slider("Passages per search", 2, 10, 5)
    if st.button("Clear workspace"):
        clear()
        st.rerun()

if uploads:
    raw = [(f.name, f.getvalue()) for f in uploads]
    fingerprint = hashlib.sha256(b"".join(n.encode() + hashlib.sha256(d).digest() for n, d in raw)).hexdigest()
    if fingerprint != st.session_state.fingerprint:
        try:
            with st.spinner("Reading and indexing PDFs..."):
                pages, warnings = read_pdfs(raw)
                chunks = split_pages(pages)
                fresh = DocumentIndex()
                try:
                    fresh.add(chunks)
                except Exception:
                    fresh.close()
                    raise
                clear()
                st.session_state.index = fresh
                st.session_state.fingerprint = fingerprint
                st.session_state.sources = sorted({page.source for page in pages})
                st.session_state.chunks = len(chunks)
            for warning in warnings:
                st.warning(warning)
        except Exception as exc:
            st.error(f"Could not index: {exc}")
elif st.session_state.index is not None:
    clear()

index = st.session_state.index
if index is None:
    st.info("Upload one or more text-based PDFs using the sidebar.")
else:
    c1, c2 = st.columns(2)
    c1.metric("Documents", len(st.session_state.sources))
    c2.metric("Indexed passages", st.session_state.chunks)

chat, eval_tab, about = st.tabs(["Chat", "Retrieval evaluation", "How it works"])
with chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("evidence"):
                with st.expander("Source passages"):
                    for i, source in enumerate(message["evidence"], 1):
                        st.markdown(f"**[S{i}] {source['source']}, page {source['page']}**")
                        st.caption(source["text"])
    question = st.chat_input("Ask a question about your PDFs", disabled=index is None)
    if question and index is not None:
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Looking for evidence..."):
                    if provider == "Search only":
                        found = index.search(question, k)
                        answer = "Closest passages from the documents:" if found else "No passages found."
                    else:
                        answer, found = ask_agent(index, question, history, provider, model, api_key, k)
                st.markdown(answer)
                evidence = [{"source": e.source, "page": e.page, "text": e.text} for e in found]
                if evidence:
                    with st.expander("Source passages", expanded=True):
                        for i, source in enumerate(evidence, 1):
                            st.markdown(f"**[S{i}] {source['source']}, page {source['page']}**")
                            st.caption(source["text"])
                st.session_state.messages.append({"role": "assistant", "content": answer, "evidence": evidence})
            except Exception as exc:
                # Provider errors can include request details. Keep credentials out of the UI.
                detail = str(exc).replace(str(api_key), "[redacted]") if api_key else str(exc)
                st.error(f"Question failed: {detail}")
    if st.session_state.messages:
        st.download_button("Export chat history", json.dumps(st.session_state.messages, indent=2), "chat.json", "application/json")

with eval_tab:
    st.write("Evaluate passage retrieval with labeled test questions. Hit@k and MRR@k do not measure answer correctness.")
    st.caption('One JSON object per line: {"question":"What was revenue?","expected_source":"report.pdf","expected_text":"revenue"}')
    pasted = st.text_area("Evaluation dataset (JSONL)", height=160)
    if st.button("Run evaluation", disabled=index is None):
        try:
            cases = [json.loads(line) for line in pasted.splitlines() if line.strip()]
            metrics, rows = evaluate_retrieval(index, cases, k)
            a, b, c = st.columns(3)
            a.metric("Questions", metrics["questions"])
            b.metric(f"Hit@{k}", f"{metrics['hit_at_k']:.1%}")
            c.metric(f"MRR@{k}", f"{metrics['mrr_at_k']:.3f}")
            st.dataframe(rows)
            st.download_button("Export results", json.dumps({"metrics": metrics, "rows": rows}, indent=2), "evaluation.json", "application/json")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

with about:
    st.markdown("PDF text → page-aware chunks → sentence embeddings → ChromaDB → CrewAI search tool → answer and excerpts. The agent can search repeatedly. PDFs must contain selectable text; scanned PDFs need OCR. This is a session-only demo; no persistent storage. Always verify citations and interpretations. Ollama only works when the server runs on the same machine.")
