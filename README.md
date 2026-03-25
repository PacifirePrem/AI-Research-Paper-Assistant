AI Research Paper Assistant

RAG-Based Intelligent Document Question Answering System

Overview

The AI Research Paper Assistant is an end-to-end application that allows users to upload research papers (PDF) and ask questions about their content.
It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers instead of generic responses.

Live Demo
https://ai-research-paper-assistant-xigb4j5vxb5dnlvgfejchj.streamlit.app/

Demo
<img width="1914" height="928" alt="image" src="https://github.com/user-attachments/assets/7d68f49f-c787-4778-bfd9-b605963ae69a" />

Key Features
Upload and process research papers (PDF)
Semantic search using embeddings
Context-aware answer generation using LLM
Fast retrieval using FAISS vector database
Interactive chat-style interface
Deployed using Streamlit Cloud

Architecture
<img width="507" height="323" alt="image" src="https://github.com/user-attachments/assets/fdb7ff4c-25e3-4104-9e78-219408f82045" />

| Category             | Tools                                |
| -------------------- | ------------------------------------ |
| Programming Language | Python                               |
| Frontend             | Streamlit                            |
| Embeddings           | Sentence Transformers                |
| Vector Database      | FAISS                                |
| LLM                  | HuggingFace Transformers / Local LLM |
| PDF Processing       | PyPDF                                |
| Deployment           | Streamlit Cloud                      |

Tech stack
- Python
- FAISS
- Sentence Transformers
- Streamlit
