import sys
import os

# Tell Python: "src is root folder"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import streamlit as st

from main import build_pipeline, answer_query

st.title("📄 AI Research Paper Assistant")

# Upload PDF
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Process Document"):
        with st.spinner("Processing paper..."):
            chunks, index = build_pipeline("temp.pdf")

        st.session_state["chunks"] = chunks
        st.session_state["index"] = index
        st.session_state["chat_history"] = []

        st.success("Document processed!")

# Chat UI
if "chunks" in st.session_state:

    st.subheader("💬 Ask questions about the paper")

    # Display previous chat
    if "chat_history" in st.session_state:
        for chat in st.session_state["chat_history"]:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])

    # Input box
    query = st.chat_input("Ask something...")

    if query:

        # Show user message
        st.chat_message("user").write(query)

        with st.spinner("Thinking..."):
            answer = answer_query(
                query,
                st.session_state["chunks"],
                st.session_state["index"]
            )

        # Show assistant response
        st.chat_message("assistant").write(answer)

        # Save conversation
        st.session_state["chat_history"].append({"role": "user", "content": query})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})