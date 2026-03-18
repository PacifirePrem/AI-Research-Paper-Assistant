import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from src.main import build_pipeline, answer_query

st.title("📄 AI Research Paper Assistant")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    if st.button("Process Document"):

        with st.spinner("Processing paper..."):
            chunks, index = build_pipeline("temp.pdf")

        st.session_state["chunks"] = chunks
        st.session_state["index"] = index

        st.success("Document processed!")

# Query section
if "chunks" in st.session_state:

    query = st.text_input("Ask a question about the paper")

    if query:

        with st.spinner("Generating answer..."):
            answer = answer_query(
                query,
                st.session_state["chunks"],
                st.session_state["index"]
            )

        st.subheader("Answer")
        st.write(answer)