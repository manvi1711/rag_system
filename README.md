RAG System with Amazon Bedrock + FAISS
Phase-1: Text-based RAG

This project implements a production-ready Retrieval-Augmented Generation (RAG) pipeline that enables users to ask natural-language questions over internal documents (PDF/DOCX/TXT) and receive:
i)Accurate answers grounded in source documents
ii)Traceable citations with page-level context
iii) Performance metrics (response time, token usage, latency)
The system uses Amazon Bedrock for both embeddings and text generation, and FAISS as the vector database for extremely fast similarity search.

Key Features:
Embeddings= Amazon Titan titan-embed-text-v1
LLM Generation=	Amazon Titan titan-text-lite (or similar)
Vector Store= 	FAISS (local index)
Document Types=	PDF, DOCX, TXT
Chunking=	RecursiveCharacter Text Splitter
CLI Interface=	Python argparse
Tracing=	Sources + Snippets + Metrics

How It Works:
        ┌─────────────┐
        │  Documents  │   (PDF/DOCX/TXT)
        └──────┬──────┘
               │  Ingest
               ▼
 ┌─────────────────────────────────┐
 │ Split into chunks (overlap)     │
 │ Embed chunks using Titan        │
 │ Build FAISS vector index        │
 │ Save index locally              │
 └─────────────────────────────────┘
               │
               │ Query
               ▼
 ┌─────────────────────────────────┐
 │ Embed question using Titan      │
 │ Retrieve top-K chunks via FAISS │
 │ Add context to LLM prompt       │
 │ Generate final answer           │
 │ Show citations + metrics        │
 └─────────────────────────────────┘

Setup:
1️) Install dependencies:
pip install -r requirements.txt

2️) Configure AWS:
~/.aws/credentials

3️) Configure environment variables:
Create .env at the project root:
AWS_REGION=us-east-1
EMBEDDING_MODEL=amazon.titan-embed-text-v1
TEXT_MODEL=amazon.titan-text-lite
DOCUMENT_PATH=documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

Step 1 — Ingest Documents
Place PDFs/DOCX/TXT inside the documents/ folder and run:
python main.py ingest

Step 2 — Query the RAG System
Example query:
python main.py query "What revenue NVIDIA announced for its fourth quarter and fiscal year?"

Design Choices & Rationale:
1) Amazon Titan embeddings: 	High-quality multilingual embeddings optimized for enterprise RAG
2) Amazon Titan text generation:	Reproducible grounded answers with low hallucination risk
3) FAISS instead of Chroma: 	Higher reliability on Windows + local runtime performance
4) Local FAISS index: 	No external dependencies & faster debugging
5) Chunking with overlap: 	Preserves context and minimizes fragmentation
6) CLI interface: Quick prototyping and easy automation

    






