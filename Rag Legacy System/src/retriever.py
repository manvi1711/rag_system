#“The retriever is a simple but robust pipeline:
# load the FAISS index, embed the user query with Titan, do a vector search, and then call Titan LLM with the retrieved context.

import json
import logging
import boto3
from botocore.config import Config

from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

from .config import (
    REGION_NAME,
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    TITAN_EMBEDDING_MODEL,
    TITAN_LLM_MODEL,
)

INDEX_PATH = "faiss_index"
logger = logging.getLogger(__name__)


def _get_bedrock_client():
    config = Config(connect_timeout=CONNECT_TIMEOUT, read_timeout=READ_TIMEOUT)
    return boto3.client("bedrock-runtime", region_name=REGION_NAME, config=config)




def run_query(question: str) -> dict:
    logger.info(f"Query received: {question}")

    client = _get_bedrock_client()
    embedder = BedrockEmbeddings(client=client, model_id=TITAN_EMBEDDING_MODEL)

    #check FAISS index
    if not Path(INDEX_PATH).exists():
        return {
            "result": "Index not found. Please run 'python main.py ingest' before querying.",
            "source_documents": [],
            "usage": {},
        }


    # Load FAISS index
    logger.info("Loading FAISS index...")
    vectorstore = FAISS.load_local(INDEX_PATH, embedder, allow_dangerous_deserialization=True)

    # Embed query
    logger.info("Embedding query...")
    query_vec = embedder.embed_query(question)

    # Perform search
    logger.info("Running FAISS similarity search...")
    docs = vectorstore.similarity_search_by_vector(query_vec, k=3)

    if not docs:
        return {"result": "No relevant documents found.", "source_documents": []}

    # Build context for the LLM
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    logger.info("Calling Titan LLM...")
    body = {
        "inputText": prompt,
        "textGenerationConfig": {"maxTokenCount": 500, "temperature": 0.1},
    }

    try:
        response = client.invoke_model(modelId=TITAN_LLM_MODEL, body=json.dumps(body))
    except Exception as e:
        logger.error(f"Bedrock LLM call failed: {e}", exc_info=True)
        return {
            "result": "The language model is temporarily unavailable. Please try again later.",
            "source_documents": docs,
            "usage": {},
        }
    # Parse body
    raw_body = response["body"].read()
    data = json.loads(raw_body)
    result = data["results"][0]["outputText"].strip()

    # Extract usage metadata if available
    headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    input_tokens = headers.get("x-amzn-bedrock-input-token-count")
    output_tokens = headers.get("x-amzn-bedrock-output-token-count")
    latency_ms = headers.get("x-amzn-bedrock-invocation-latency")

    def _to_int(maybe_str):
        try:
            return int(maybe_str)
        except Exception:
            return None

    usage = {
        "input_tokens": _to_int(input_tokens),
        "output_tokens": _to_int(output_tokens),
        "latency_ms": _to_int(latency_ms),
    }

    return {
        "result": result,
        "source_documents": docs,
        "usage": usage,
    }
