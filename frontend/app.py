# frontend/app.py
import streamlit as st
import os
st.sidebar.caption(f"API: {os.getenv('API_BASE_URL', 'NOT SET')}")

from api_client import (
    ask_question,
    check_health,
    delete_document,
    list_documents,
    upload_document,
)

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ─────────────────────────────────────────────
# These persist across reruns within the same session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, citations}

if "documents" not in st.session_state:
    st.session_state.documents = []

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = "All documents"


# ── Helper: refresh document list from backend ────────────────────────────────
def refresh_documents():
    result = list_documents()
    if result["success"]:
        st.session_state.documents = result["data"]["documents"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Doc Intelligence")
    st.caption("Powered by Groq + Qdrant + sentence-transformers")

    st.divider()

    # API health indicator
    is_healthy = check_health()
    if is_healthy:
        st.success("API connected", icon="🟢")
    else:
        st.error("API unreachable", icon="🔴")
        st.caption("Start your FastAPI server: `uvicorn app.main:app --reload`")

    st.divider()

    # --- Document upload ---
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Maximum file size: 20MB",
    )

    if uploaded_file is not None:
        if st.button("Ingest Document", use_container_width=True, type="primary"):
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                result = upload_document(
                    file_bytes=uploaded_file.read(),
                    filename=uploaded_file.name,
                )
            if result["success"]:
                data = result["data"]
                st.success(
                    f"✅ '{data['filename']}' ingested\n\n"
                    f"{data['chunks_stored']} chunks stored"
                )
                refresh_documents()
            else:
                st.error(f"❌ {result['error']}")

    st.divider()

    # --- Document manager ---
    st.subheader("Manage Documents")
    refresh_documents()

    if not st.session_state.documents:
        st.info("No documents uploaded yet.")
    else:
        for doc in st.session_state.documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📎 {doc}")
            with col2:
                if st.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                    result = delete_document(doc)
                    if result["success"]:
                        st.success(f"Deleted '{doc}'")
                        refresh_documents()
                        st.rerun()
                    else:
                        st.error(result["error"])

    st.divider()

    # --- Search scope selector ---
    st.subheader("Search Scope")
    doc_options = ["All documents"] + st.session_state.documents
    st.session_state.selected_doc = st.selectbox(
        "Ask questions about:",
        options=doc_options,
        index=0,
        help="Choose a specific document or search across all uploaded files",
    )

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=15,
        value=5,
        help="More chunks = more context but slower responses",
    )

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Document Intelligence RAG System")
st.caption(
    "Upload PDF documents, then ask questions. "
    "Every answer includes citations with page numbers."
)

if not is_healthy:
    st.warning(
        "⚠️ The backend API is not reachable. "
        "Please start the FastAPI server before asking questions.",
        icon="⚠️",
    )

st.divider()

# ── Chat history display ───────────────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show citations only for assistant messages that have them
        if message["role"] == "assistant" and message.get("citations"):
            with st.expander(
                f"📚 Sources ({len(message['citations'])} citation(s))",
                expanded=False,
            ):
                for i, citation in enumerate(message["citations"], start=1):
                    st.markdown(f"**[{i}] {citation['source']} — Page {citation['page_number']}**")
                    st.caption(f"> {citation['relevant_excerpt']}")
                    if i < len(message["citations"]):
                        st.divider()

            # Confidence badge
            confidence = message.get("confidence", "")
            badge_colour = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
            st.caption(
                f"{badge_colour} Confidence: **{confidence}** | "
                f"Chunks retrieved: **{message.get('chunks_retrieved', '—')}**"
            )


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=not is_healthy,
):
    # Add user message to history and display immediately
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to API and display response
    with st.chat_message("assistant"):
        source_filter = (
            None
            if st.session_state.selected_doc == "All documents"
            else st.session_state.selected_doc
        )

        with st.spinner("Searching documents and generating answer..."):
            result = ask_question(
                question=prompt,
                source_filter=source_filter,
                top_k=top_k,
            )

        if result["success"]:
            data = result["data"]
            answer = data["answer"]
            citations = data["citations"]
            confidence = data["confidence"]
            chunks_retrieved = data["chunks_retrieved"]

            st.markdown(answer)

            if citations:
                with st.expander(
                    f"📚 Sources ({len(citations)} citation(s))",
                    expanded=True,
                ):
                    for i, citation in enumerate(citations, start=1):
                        st.markdown(
                            f"**[{i}] {citation['source']} — Page {citation['page_number']}**"
                        )
                        st.caption(f"> {citation['relevant_excerpt']}")
                        if i < len(citations):
                            st.divider()

            badge_colour = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
            st.caption(
                f"{badge_colour} Confidence: **{confidence}** | "
                f"Chunks retrieved: **{chunks_retrieved}**"
            )

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "confidence": confidence,
                "chunks_retrieved": chunks_retrieved,
            })

        else:
            error_msg = f"❌ {result['error']}"
            st.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "citations": [],
            })