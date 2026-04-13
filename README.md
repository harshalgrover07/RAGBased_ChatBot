# RAG Document Chatbot — AI-Powered Q&A over PDFs

## Overview
A production-ready Retrieval-Augmented Generation (RAG) chatbot that enables semantic question-answering over any PDF document. Built using HuggingFace Transformer embeddings, FAISS vector search, and Google Gemini LLM.

## Tech Stack
- **LLM**: Google Gemini API
- **Embeddings**: HuggingFace sentence-transformers
- **Vector Store**: FAISS
- **Framework**: LangChain / custom RAG pipeline
- **UI**: Streamlit
- **Language**: Python

## Architecture
PDF Input → Text Extraction → Chunking → HuggingFace Embeddings
→ FAISS Vector Index → Query Retrieval → Gemini LLM → Answer

## Features
- Upload any PDF and ask natural language questions
- Context-aware responses grounded in document content
- Session-based chat history
- Fast semantic retrieval using FAISS

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Results
Accurate semantic retrieval and context-aware answers across multi-page PDFs.
