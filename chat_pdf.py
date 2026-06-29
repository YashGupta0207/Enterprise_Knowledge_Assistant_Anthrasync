"""
chat_pdf.py — Streamlit UI for the Enterprise Knowledge Assistant.

Run with:  streamlit run chat_pdf.py

All retrieval/generation logic lives in rag_core.py; this file is
responsible for the UI only: file upload, indexing controls, chat history,
and rendering answers/sources/confidence.
"""

import logging
import os

import streamlit as st

import rag_core
from rag_core import (
    EmptyDocumentError,
    IndexNotFoundError,
    InvalidQuestionError,
    LLMConfigError,
    LLMGenerationError,
    RAGError,
)

rag_core.configure_logging()
log = logging.getLogger("eka.streamlit")


# ─────────────────────────── CACHED RESOURCES ───────────────── #

@st.cache_resource(show_spinner=False)
def cached_embeddings():
    return rag_core.get_embeddings()


# ─────────────────────────── STREAMLIT UI ───────────────────── #

def main():
    st.set_page_config(
        page_title="Enterprise Knowledge Assistant",
        page_icon="🏢",
        layout="wide",
    )

    # Warm the embeddings cache through Streamlit's resource cache so it
    # survives reruns (rag_core also caches it module-globally as a backstop).
    cached_embeddings()

    with st.sidebar:
        st.title("📂 Document Upload")

        indexed_files = rag_core.load_indexed_files()
        index_ready = rag_core.index_exists()

        if index_ready and not st.session_state.get("docs_processed"):
            st.session_state["docs_processed"] = True

        if indexed_files:
            st.markdown("#### 📚 Already indexed:")
            for name in sorted(indexed_files):
                st.markdown(f"- ✅ `{name}`")
            st.markdown("---")

        st.markdown("Upload new PDF documents below.")

        pdf_docs = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        new_files, already_in = [], []

        if pdf_docs:
            for pdf in pdf_docs:
                if pdf.name in indexed_files:
                    already_in.append(pdf.name)
                else:
                    new_files.append(pdf)

            if already_in:
                st.warning(
                    "⚠️ Already indexed (will be skipped):\n"
                    + "\n".join(f"- `{n}`" for n in already_in)
                )
            if new_files:
                st.info(
                    "🆕 New — will be added to index:\n"
                    + "\n".join(f"- `{f.name}`" for f in new_files)
                )

        btn_disabled = not bool(new_files)
        btn_label = (
            "⚙️ Add New Documents to Index"
            if new_files else
            ("✅ All uploaded files already indexed" if already_in else "⚙️ Process Documents")
        )

        if st.button(btn_label, use_container_width=True, disabled=btn_disabled):
            with st.spinner("Extracting text and building index…"):
                try:
                    pages = rag_core.extract_text_with_metadata(new_files)
                    if not pages:
                        st.error(
                            "❌ No readable text found. "
                            "The PDF may be image-only (scanned) or corrupted. "
                            "Try a text-based PDF, or run OCR on it first."
                        )
                    else:
                        texts, metadatas = rag_core.chunk_pages(pages)

                        if index_ready:
                            rag_core.append_to_vector_store(texts, metadatas)
                        else:
                            rag_core.build_vector_store(texts, metadatas)

                        indexed_files.update(f.name for f in new_files)
                        rag_core.save_indexed_files(indexed_files)

                        st.session_state["docs_processed"] = True
                        st.success(
                            f"✅ Added {len(texts)} chunks "
                            f"from {len(new_files)} new document(s)."
                        )
                        st.rerun()
                except Exception as e:
                    log.exception("Indexing failed")
                    st.error(f"❌ Indexing failed: {e}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()
        with col2:
            if st.button("🔄 Reset Index", use_container_width=True):
                import shutil
                if os.path.exists(rag_core.FAISS_INDEX_PATH):
                    shutil.rmtree(rag_core.FAISS_INDEX_PATH)
                st.session_state["docs_processed"] = False
                st.session_state["messages"] = []
                st.success("Index cleared.")
                st.rerun()

        st.markdown("---")
        with st.expander("⚙️ Retrieval settings (read-only)"):
            st.caption(f"Dense candidates (FAISS): {rag_core.TOP_K_DENSE}")
            st.caption(f"Sparse candidates (BM25): {rag_core.TOP_K_SPARSE}")
            st.caption(f"After RRF fusion: {rag_core.TOP_K_FUSED}")
            st.caption(f"After MMR rerank → sent to LLM: {rag_core.TOP_K_FINAL}")
            st.caption(f"LLM model: {rag_core.LLM_MODEL}")

    # ── Main panel ──
    st.title("🏢 Enterprise Knowledge Assistant")
    st.caption(
        "Ask questions about your internal documents. "
        "Answers cite the source document and page, "
        "using hybrid (keyword + semantic) retrieval with diversity reranking."
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s['document']}** — Page {s['page']}")
            if msg.get("confidence") and msg["role"] == "assistant":
                st.caption(f"Confidence: {msg['confidence']:.0%}")

    if question := st.chat_input("Ask a question about your documents…"):
        if not rag_core.index_exists():
            st.warning("⚠️ Please upload and process documents first.")
        else:
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                answer, sources, confidence = "", [], 0.0
                with st.spinner("Searching documents…"):
                    try:
                        result = rag_core.answer_question(
                            question, st.session_state["messages"][:-1]
                        )
                        answer = result.answer
                        sources = result.sources
                        confidence = result.confidence
                    except InvalidQuestionError as e:
                        answer = f"⚠️ {e}"
                    except IndexNotFoundError as e:
                        answer = f"⚠️ {e}"
                    except LLMConfigError as e:
                        answer = f"❌ Configuration error: {e}"
                    except LLMGenerationError as e:
                        answer = f"❌ The model failed to respond: {e}"
                    except EmptyDocumentError as e:
                        answer = f"⚠️ {e}"
                    except RAGError as e:
                        answer = f"❌ Something went wrong: {e}"
                    except Exception as e:
                        log.exception("Unexpected error answering question")
                        answer = "❌ An unexpected error occurred. Please try again."

                st.markdown(answer)
                if sources:
                    with st.expander("📎 Sources"):
                        for s in sources:
                            st.markdown(f"- **{s['document']}** — Page {s['page']}")
                if confidence:
                    st.caption(f"Confidence: {confidence:.0%}")

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence,
            })


if __name__ == "__main__":
    main()
