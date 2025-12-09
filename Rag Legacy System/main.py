#main.py is just a thin CLI wrapper. It defines two sub-commands:
#ingest= load docs, chunk, embed and build the FAISS index;
#query= take a user question, call the retrieval + generation pipeline, and print answer, sources and metrics.
#Uses argparse to define ingest and query
#For ingest, it calls load_and_split_documents() then create_and_store_embeddings()
#For query, it calls run_query(question)

import argparse
import logging
import sys

from src.ingestor import load_and_split_documents, create_and_store_embeddings
from src.retriever import run_query

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents and store embeddings")

    query_parser = subparsers.add_parser("query", help="Ask question against the knowledge base")
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

        #metrics
        print("\n--- Metrics ---\n")
        print(f"Response time: {elapsed:.2f} seconds")

        usage = response.get("usage", {})
        if usage:
            print(f"Input tokens:  {usage.get('input_tokens')}")
            print(f"Output tokens: {usage.get('output_tokens')}")
            print(f"Latency:       {usage.get('latency_ms')} ms")

        print()
        return

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
