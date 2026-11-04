import sys
import os
import hashlib
import tempfile

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    ),
)

import streamlit as st

from main import build_pipeline, answer_query


st.title("📄 AI Research Paper Assistant")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a research paper (PDF)",
    type=["pdf"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    document_id = hashlib.sha256(file_bytes).hexdigest()

    # Reset the previous document's state when a new PDF is selected.
    if st.session_state.get("document_id") != document_id:
        st.session_state.pop("chunks", None)
        st.session_state.pop("index", None)
        st.session_state.pop("chat_history", None)

        st.session_state["document_id"] = document_id
        st.session_state["processed_document_id"] = None

    # Process the uploaded document.
    if st.button("Process Document"):
        temp_path = None

        try:
            with st.spinner("Processing paper..."):
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False,
                ) as temp_file:
                    temp_file.write(file_bytes)
                    temp_path = temp_file.name

                try:
                    chunks, index = build_pipeline(temp_path)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["processed_document_id"] = document_id
            st.session_state["chat_history"] = []

            st.success("Document processed successfully!")

        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")

# Show chat only after the currently selected PDF has been processed.
if (
    uploaded_file is not None
    and st.session_state.get("processed_document_id")
    == st.session_state.get("document_id")
    and "chunks" in st.session_state
    and "index" in st.session_state
):
    st.subheader("💬 Ask questions about the paper")

    # Display previous messages and their retrieved source passages.
    for chat in st.session_state.get("chat_history", []):
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

            if chat["role"] == "assistant" and chat.get("sources"):
                sources = chat["sources"]

                # Backward compatibility with older page-number-only history.
                if sources and isinstance(sources[0], int):
                    pages = ", ".join(
                        f"Page {page}" for page in sources
                    )
                    st.caption(f"Retrieved sources: {pages}")

                else:
                    pages = list(dict.fromkeys(
                        source["page"] for source in sources
                    ))

                    st.caption(
                        "Retrieved sources: "
                        + ", ".join(f"Page {page}" for page in pages)
                    )

                    with st.expander("📚 Sources used"):
                        for i, source in enumerate(sources, start=1):
                            st.markdown(
                                f"**Source {i} — Page {source['page']}**"
                            )
                            st.write(source["text"])
                            st.divider()

    # Question input
    query = st.chat_input("Ask something...")

    if query:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": query,
        })

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Finding relevant passages..."):
                answer, retrieved_chunks = answer_query(
                    query,
                    st.session_state["chunks"],
                    st.session_state["index"],
                )

            st.markdown(answer)

            if retrieved_chunks:
                source_pages = list(dict.fromkeys(
                    source["page"] for source in retrieved_chunks
                ))

                st.caption(
                    "Retrieved sources: "
                    + ", ".join(
                        f"Page {page}" for page in source_pages
                    )
                )

                with st.expander("📚 Sources used"):
                    for i, source in enumerate(
                        retrieved_chunks, start=1
                    ):
                        st.markdown(
                            f"**Source {i} — Page {source['page']}**"
                        )
                        st.write(source["text"])
                        st.divider()

            else:
                st.info("No source passages were retrieved.")

        # Save the answer and its supporting retrieved passages.
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks,
        })
