"""Streamlit web UI.   streamlit run app.py

Pages: Chat | Documents | Retrieval Debug | Evaluation
"""
import json
import logging
import uuid
from pathlib import Path

import streamlit as st

from rag.citations import format_citation
from rag.config import DOCS_DIR, EMBED_MODEL, EVAL_DATASET, LLM_MODEL, LLM_PROVIDER, RERANKER_MODEL
from rag.evaluation import load_dataset, run_evaluation
from rag.ingestion import validate_upload
from rag.llm import get_llm
from rag.logging_config import setup_logging
from rag.security import validate_query
from rag.system import build_system
from rag.vectorstore import delete_document, index_file, list_documents

setup_logging()
logger = logging.getLogger("app")
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")
DOCS_DIR.mkdir(exist_ok=True)


@st.cache_resource(show_spinner="Loading models and indexing documents...")
def load_backend():
    return build_system()


vectorstore, retriever, graph = load_backend()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


def chat_markdown() -> str:
    lines = ["# Conversation", ""]
    for m in st.session_state.messages:
        lines.append(f"**{'You' if m['role'] == 'user' else 'Bot'}:** {m['content']}\n")
        if m.get("sources"):
            lines.append("Sources: " + "; ".join(m["sources"]) + "\n")
    return "\n".join(lines)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📚 RAG Chatbot")
    page = st.radio("Page", ["Chat", "Documents", "Retrieval Debug", "Evaluation"], label_visibility="collapsed")
    st.divider()
    docs_now = list_documents(vectorstore)
    st.caption(f"LLM: `{LLM_PROVIDER}` / `{LLM_MODEL}`")
    st.caption(f"Embeddings: `{EMBED_MODEL.split('/')[-1]}`")
    st.caption(f"Reranker: `{RERANKER_MODEL.split('/')[-1]}`" if retriever.reranker else "Reranker: off")
    st.caption(f"{len(docs_now)} documents · {sum(docs_now.values())} chunks")
    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    if st.session_state.messages:
        st.download_button("Export chat (.md)", chat_markdown(), file_name="conversation.md")


# ---------- Page: Chat ----------
def page_chat():
    st.header("Chat")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                st.caption("Sources: " + "  ·  ".join(m["sources"]))
            for f in m.get("flags", []):
                st.warning(f"🛡️ {f}")

    raw = st.chat_input("Ask something about your documents...")
    if not raw:
        return
    try:
        prompt = validate_query(raw)
    except ValueError as e:
        st.warning(str(e))
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    def token_stream():
        for chunk, meta in graph.stream({"messages": [("user", prompt)]}, config, stream_mode="messages"):
            if meta.get("langgraph_node") == "generate" and isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

    with st.chat_message("assistant"):
        box = st.empty()
        sources, flags = [], []
        try:
            with box.container():
                streamed = st.write_stream(token_stream())
            values = graph.get_state(config).values
            answer = values["messages"][-1].content
            if answer != streamed:            # not-found message, or the security layer edited the answer
                box.markdown(answer)
            ctx = values.get("context", [])
            info = values.get("retrieval", {})
            flags = values.get("security_flags", [])
            sources = [f"[{i}] {format_citation(d)}" for i, d in enumerate(ctx, 1)]
            if sources:
                st.caption("Sources: " + "  ·  ".join(sources))
            for f in flags:
                st.warning(f"🛡️ {f}")
            gen = values.get("generation_ms")
            st.caption(
                f"Retrieval {info.get('elapsed_ms', 0):.0f} ms"
                + (f" · Generation {gen / 1000:.1f} s" if gen else "")
                + f" · {info.get('gate_type', '')} relevance {info.get('gate_score', 0)}"
            )
            with st.expander("Retrieved passages"):
                st.caption(f"Searched for: {values.get('standalone_question', prompt)}")
                for i, d in enumerate(ctx, 1):
                    st.markdown(f"**[{i}] {format_citation(d)}**")
                    st.text(d.page_content[:500])
        except Exception:
            logger.exception("Chat failed")
            answer = "Sorry, something went wrong. Check that your LLM (Ollama / Groq) is running, then try again."
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "flags": flags})


# ---------- Page: Documents ----------
def page_documents():
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads and st.button("Add to knowledge base", type="primary"):
        with st.spinner("Indexing..."):
            for up in uploads:
                path = None
                try:
                    name = validate_upload(up.name, up.size, bytes(up.getbuffer()[:8]))
                    path = DOCS_DIR / name
                    path.write_bytes(up.getbuffer())
                    status, n = index_file(vectorstore, path)
                    st.success({
                        "added": f"{name}: {n} chunks added",
                        "updated": f"{name}: content changed, re-indexed ({n} chunks)",
                        "skipped": f"{name}: unchanged, already indexed",
                    }[status])
                except Exception as e:
                    if path is not None:
                        delete_document(vectorstore, str(path))   # don't leave a broken file behind
                    if isinstance(e, ValueError):
                        st.error(f"{up.name}: {e}")
                    else:
                        logger.exception("Failed to process %s", up.name)
                        st.error(f"{up.name}: could not be processed (see logs/rag.log).")

    st.subheader("Indexed documents")
    docs = list_documents(vectorstore)
    if not docs:
        st.info("No documents yet. Upload one above.")
    for source, n_chunks in docs.items():
        c1, c2 = st.columns([8, 1])
        c1.markdown(f"**{Path(source).name}** — {n_chunks} chunks")
        if c2.button("🗑️", key=f"del_{source}", help="Delete this document"):
            delete_document(vectorstore, source)
            st.rerun()


# ---------- Page: Retrieval Debug ----------
def page_debug():
    st.header("Retrieval Debug")
    st.caption("See exactly what the retriever finds for a query, with every score. No LLM is called.")
    q = st.text_input("Query", placeholder="e.g. What is Kerberos?")
    if not q.strip():
        return
    with st.spinner("Retrieving..."):
        result = retriever.retrieve(q.strip())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relevant?", "Yes" if result.relevant else "No → 'not found'")
    c2.metric(f"Gate score ({result.gate_type})", f"{result.gate_score:.3f}")
    c3.metric("Threshold", f"{result.threshold}")
    c4.metric("Retrieval time", f"{result.elapsed_ms:.0f} ms")

    kept_ids = {id(c) for c in result.candidates}
    rows = [{
        "rank": i, "sent to LLM": "✅" if id(c) in kept_ids else "",
        "citation": format_citation(c.doc), "dense (cosine)": round(c.dense, 3),
        "BM25": round(c.bm25, 2), "fused (RRF)": round(c.fused, 4),
        "rerank": None if c.rerank is None else round(c.rerank, 3),
    } for i, c in enumerate(result.considered, 1)]
    st.dataframe(rows)
    for i, c in enumerate(result.considered[:8], 1):
        with st.expander(f"{i}. {format_citation(c.doc)}"):
            st.text(c.doc.page_content[:1200])


# ---------- Page: Evaluation ----------
def page_eval():
    st.header("Evaluation")
    st.caption("Measure retrieval and answer quality on a question set (see evaluation/dataset.json).")
    uploaded = st.file_uploader("Dataset (JSON) — optional, defaults to evaluation/dataset.json", type=["json"])
    c1, c2 = st.columns(2)
    do_generate = c1.checkbox("Also generate answers (slow)")
    do_judge = c2.checkbox("LLM-judge faithfulness (slower, noisy)", disabled=not do_generate)

    if st.button("Run evaluation", type="primary"):
        try:
            data = json.loads(uploaded.getvalue()) if uploaded else None
            dataset = (data["questions"] if isinstance(data, dict) else data) if data else load_dataset(EVAL_DATASET)
        except Exception as e:
            st.error(f"Could not read the dataset: {e}")
            return
        indexed = {Path(s).name for s in list_documents(vectorstore)}
        with st.spinner(f"Evaluating {len(dataset)} questions..."):
            st.session_state.eval_report = run_evaluation(
                retriever, dataset, graph=graph if do_generate else None,
                judge_llm=get_llm() if (do_generate and do_judge) else None, indexed_sources=indexed,
            )

    report = st.session_state.get("eval_report")
    if not report:
        return
    for w in report["warnings"]:
        st.warning(w)
    summary = {k: v for k, v in report["summary"].items() if v is not None}
    cols = st.columns(4)
    for i, (k, v) in enumerate(summary.items()):
        cols[i % 4].metric(k.replace("_", " "), v)
    st.dataframe(report["rows"])


{"Chat": page_chat, "Documents": page_documents, "Retrieval Debug": page_debug, "Evaluation": page_eval}[page]()
