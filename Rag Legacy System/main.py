#main.py is just a thin CLI wrapper. It defines two sub-commands:
#ingest= load docs, chunk, embed and build the FAISS index;
#query= take a user question, call the retrieval + generation pipeline, and print answer, sources and metrics.
#Uses argparse to define ingest and query
#For ingest, it calls load_and_split_documents() then create_and_store_embeddings()
#For query, it calls run_query(question)

import argparse
import logging
import sys
import os
import json
from datetime import datetime

from src.ingestor import load_and_split_documents, create_and_store_embeddings
from src.retriever import run_query

logger = logging.getLogger(__name__)

# --- query logging config ---
LOGS_DIR = "logs"
QUERIES_LOG_PATH = os.path.join(LOGS_DIR, "queries.jsonl")


def log_query_to_jsonl(question: str, response: dict, elapsed_seconds: float) -> None:
    """
    Append a single query + response record to logs/queries.jsonl
    in JSONL format (one JSON object per line).
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Extract answer
    answer = response.get("result")

    # Extract sources
    sources = []
    for doc in response.get("source_documents", []) or []:
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source", "Unknown")
        page = meta.get("page", None)
        sources.append(
            {
                "source": src,
                "page": page,
            }
        )

    # Extract metrics / usage
    usage = response.get("usage", {}) or {}
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "answer": answer,
        "sources": sources,
        "metrics": {
            "response_time_seconds": round(elapsed_seconds, 3),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "latency_ms": usage.get("latency_ms"),
        },
    }

    
    with open(QUERIES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command")

    _ = subparsers.add_parser("ingest", help="Ingest documents and store embeddings")

    query_parser = subparsers.add_parser(
        "query", help="Ask question against the knowledge base"
    )
    query_parser.add_argument("question", type=str, help="Question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        print("DEBUG: entered ingest block")
        chunks = load_and_split_documents()
        if not chunks:
            print("No documents found to ingest.")
            sys.exit(1)

        create_and_store_embeddings(chunks)
        print("Ingestion completed.")
        return

    elif args.command == "query":
        print("DEBUG: entered query block")

        question = args.question
        print("DEBUG: question =", question)

        import time

        start_time = time.perf_counter()
        response = run_query(question)
        elapsed = time.perf_counter() - start_time

        # ---- answer ----
        print("\n--- Answer ---\n")
        print(response.get("result", response))

        # ---- sources ----
        print("\n--- Sources ---\n")
        sources = response.get("source_documents", [])
        if not sources:
            print("No relevant sources found.")
        else:
            for i, doc in enumerate(sources, start=1):
                meta = doc.metadata or {}
                src = meta.get("source", "Unknown")
                page = meta.get("page", None)
                snippet = doc.page_content[:200].replace("\n", " ")

                if page:
                    print(f"[{i}] {src} (page {page})")
                else:
                    print(f"[{i}] {src}")

                print(f"     {snippet}...")

        # ---- metrics ----
        print("\n--- Metrics ---\n")
        print(f"Response time: {elapsed:.2f} seconds")

        usage = response.get("usage", {})
        if usage:
            print(f"Input tokens:  {usage.get('input_tokens')}")
            print(f"Output tokens: {usage.get('output_tokens')}")
            print(f"Latency:       {usage.get('latency_ms')} ms")

        if isinstance(response, dict):
            try:
                log_query_to_jsonl(question, response, elapsed)
            except Exception as e:
                print(f"(Warning: failed to log query: {e})")

        print()
        return

    else:
        parser.print_help()


if __name__ == "__main__":
    main()


Updated main.py — added JSONL logging and metrics
