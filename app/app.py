import streamlit as st

st.title("AI Research Paper Assistant")

st.write("Ask questions about your research papers.")

query = st.text_input("Enter your question:")

if query:
    st.write("You asked:", query)