Hybrid Multimodal RAG System (Text + OCR Images)

A modular Retrieval-Augmented Generation system that supports PDF, DOCX, TXT, and Images (with OCR).
Built using FAISS, BM25, Amazon Bedrock Titan embeddings, and Claude reranking.

 Features

Multimodal ingestion= PDF, DOCX, TXT, PNG/JPG

OCR support= EasyOCR extracts text from images

Hybrid Retrieval= FAISS (semantic) + metadata

Claude Reranking= Improves relevance

Chunking with metadata (file name, page number, chunk index)

Local vectorstore persistence

Project Structure
rag_system/
│
├── documents/         # Add your files here
├── vectorstore/       
│
├── src/
│   ├── ingestor.py        # Reads + processes files
│   ├── retriever.py       # Hybrid retrieval + reranking
│   ├── bedrock_embedder.py
│   ├── utilities.py
│   ├── config.py
│   └── __init__.py
│
└── main.py             # CLI: ingest / query

 Setup
1. Install dependencies
pip install -r requirements.txt

2. Add AWS credentials to .env
AWS_ACCESS_KEY_ID=xxxx
AWS_SECRET_ACCESS_KEY=xxxx
AWS_REGION=us-east-1

 Usage
1. Ingest documents

Add files to the documents/ folder, then run:

python main.py ingest

2. Ask questions
python main.py query "Your question here"

Example Output
Found 30 candidate documents
Reranking with Claude...
Selected 5 diverse documents

--- Answer ---
<final answer>

--- Sources ---
file.pdf - page 2
image.png (OCR)

