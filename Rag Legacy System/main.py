import sys
import logging
from src.ingestor import ingest_document
from src.retriever import retrieve_answer

# LOGGING CONFIG
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


# INGESTION PIPELINE
def run_ingestion():
    logger.info("Starting ingestion pipeline...")
    ingest_document(None)
    logger.info("Ingestion completed.")


# QUERY PIPELINE
def run_query(question: str):
    logger.info("Running query...")

    result = retrieve_answer(question)

    #ANSWER
    print("\n--- Answer ---\n")
    print(result.get("result", "No answer produced."))

    # SOURCES 
    print("\n--- Sources ---\n")
    sources = result.get("source_documents", [])

    if not sources:
        print("No relevant sources found.")
    else:
        for i, doc in enumerate(sources, start=1):
            meta = doc.metadata
            src = meta.get("source", "Unknown")
            page = meta.get("page", "?")
            snippet = doc.page_content[:250].replace("\n", " ")
            print(f"[{i}] {src} (page {page})")
            print(f"   {snippet}...\n")

    #METRICS
    print("\n--- Metrics ---\n")
    usage = result.get("usage", {})
    if usage:
        for key, val in usage.items():
            print(f"{key}: {val}")
    else:
        print("No usage metrics available.")


# COMMAND HANDLING
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py ingest")
        print("  python main.py query \"your question\"")
        sys.exit(1)

    command = sys.argv[1].lower()

    # INGEST
    if command == "ingest":
        print("Ingestion started...")
        run_ingestion()
        print("Ingestion completed.")

    # QUERY
    elif command == "query":
        if len(sys.argv) < 3:
            print("Please enter a query. Example:")
            print('python main.py query "What is NLP?"')
            sys.exit(1)

        question = " ".join(sys.argv[2:])
        run_query(question)

    # INVALID
    else:
        print(f"Unknown command: {command}")
        print("Use 'ingest' or 'query'.")
