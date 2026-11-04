# AI Research Paper Assistant

 RAG-Based Intelligent Document Question Answering System

## Overview

- The AI Research Paper Assistant is an end-to-end application that allows users to upload research papers (PDF) and ask questions about their content.
- It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers instead of generic responses.

## Live Demo
- https://ai-research-paper-assistant-xigb4j5vxb5dnlvgfejchj.streamlit.app/

## Demo

<img width="1914" height="928" alt="image" src="https://github.com/user-attachments/assets/7d68f49f-c787-4778-bfd9-b605963ae69a" />

## Key Features

- Upload and process research papers (PDF)
- Semantic search using embeddings
- Context-aware answer generation using LLM
- Fast retrieval using FAISS vector database
- Interactive chat-style interface
- Deployed using Streamlit Cloud

## Architecture

<img width="507" height="323" alt="image" src="https://github.com/user-attachments/assets/fdb7ff4c-25e3-4104-9e78-219408f82045" />

## Tech Stack

| Category             | Tools                                |
| -------------------- | ------------------------------------ |
| Programming Language | Python                               |
| Frontend             | Streamlit                            |
| Embeddings           | Sentence Transformers                |
| Vector Database      | FAISS                                |
| LLM                  | HuggingFace Transformers / Local LLM |
| PDF Processing       | PyPDF                                |
| Deployment           | Streamlit Cloud                      |

## Project Structure

<img width="497" height="361" alt="image" src="https://github.com/user-attachments/assets/e48c5602-4c91-42b2-94bd-e40cd945351e" />

## How it Works

1. Upload a research paper
2. Text is extracted and split into chunks
3. Chunks are converted into vector embeddings
4. Embeddings are stored in FAISS
5. Relevant chunks are retrieved for a query
6. LLM generates a context-aware answer
7. Answer is displayed in the chat interface

## Performance Optimizations

- Embedding caching to avoid recomputation
- Chunk size tuning for better retrieval accuracy
- Top-K retrieval for relevant context selection

## Future Improvements

- Multi-document support
- Conversation memory (chat history awareness)
- Citations in answers
- Advanced LLM integration (LLaMA / OpenAI)
- API backend using FastAPI

## Author
Prem Kumar
