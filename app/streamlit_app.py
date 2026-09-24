"""Streamlit UI: upload a PDF, ask questions, and see cited sources."""
import sys
from pathlib import Path

import streamlit as st

# Allow running as a plain script (streamlit run app/streamlit_app.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, ingest, rag  # noqa: E402

st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="centered")

st.title("📄 PDF Q&A")
st.caption("Upload a PDF, then ask questions about it. Answers cite the pages they came from.")


# ---------------------------------------------------------------- resources
@st.cache_resource
def get_vectorstore():
    return ingest.get_vectorstore()


def already_processed(file_key: str) -> bool:
    return file_key in st.session_state.processed


# ---------------------------------------------------------------- session state
if "processed" not in st.session_state:
    st.session_state.processed = {}  # file_key -> {"source":..., "pages":..., "chunks":...}
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history: {"role", "content", "sources"}


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📤 Upload PDF")

    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded is not None:
        file_key = f"{uploaded.name}:{uploaded.size}"
        if not already_processed(file_key):
            with st.spinner(f"Indexing {uploaded.name}..."):
                tmp_path = config.UPLOAD_DIR / uploaded.name
                tmp_path.write_bytes(uploaded.getvalue())
                try:
                    stats = ingest.ingest_pdf(tmp_path)
                    st.session_state.processed[file_key] = stats
                    st.success(
                        f"Indexed {stats['source']}: {stats['pages']} pages → "
                        f"{stats['chunks']} chunks"
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to index {uploaded.name}: {exc}")

    docs = st.session_state.processed
    if docs:
        st.subheader("Indexed documents")
        for key, stats in docs.items():
            st.markdown(f"- **{stats['source']}** — {stats['pages']} pages, {stats['chunks']} chunks")

        if st.button("🗑️ Clear everything", use_container_width=True):
            ingest.reset_collection()
            st.cache_resource.clear()
            st.session_state.processed = {}
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No documents indexed yet.")

    st.divider()
    st.caption(f"Chat model: `{config.CHAT_MODEL}`\n\nEmbeddings: `{config.EMBEDDING_MODEL}`")


# ---------------------------------------------------------------- chat UI
def render_sources(sources: list) -> None:
    """Show the retrieved chunks behind an answer as expandable cards."""
    with st.expander("🔍 Sources used for this answer"):
        for i, doc in enumerate(sources, start=1):
            page = int(doc.metadata.get("page", 0)) + 1
            src = doc.metadata.get("source", "unknown")
            st.markdown(f"**{i}. {src} — page {page}**")
            st.text(doc.page_content[:600])
            st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

if question := st.chat_input("Ask something about the uploaded PDF..."):
    if not st.session_state.processed:
        st.warning("Upload a PDF first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag.ask(question)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                st.stop()
        st.markdown(result["answer"])
        render_sources(result["source_docs"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["source_docs"],
        }
    )
