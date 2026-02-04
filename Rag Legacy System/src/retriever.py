# Hybrid retrieval + reranking + answer generation

import logging
from pathlib import Path

import boto3
from botocore.config import Config

from langchain_community.vectorstores import FAISS

from src.config import (
    REGION_NAME,
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    BEDROCK_EMBED_MODEL,
    BEDROCK_LLM_MODEL,
)

from .bedrock_embedder import TitanEmbedder  

from .utilities import rerank_chunks_scored, apply_page_diversity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_TOP_K = 30
MAX_CONTEXT_CHARS = 12000
MAX_CHUNK_CHARS = 1500
INDEX_PATH = "faiss_index"  # FIXED: Match ingestor's save path
REFUSAL_MESSAGE = "The document does not contain this information."


# Bedrock client
def _get_bedrock_client():
    config = Config(
        connect_timeout=CONNECT_TIMEOUT,
        read_timeout=READ_TIMEOUT,
    )
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION_NAME,
        config=config,
    )


# Converse API helper
def _invoke_llm_converse(client, model_id, prompt):
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 500, "temperature": 0.0}  # Added config
    )

    output = response["output"]["message"]["content"]
    text = output[0].get("text", "").strip()

    usage = response.get("usage", {})

    return {"text": text, "usage": usage}


# Section guesser
def guess_section(question: str):
    q = question.lower()

    mappings = {
        "network": "network",
        "protocol": "protocol",
        "tcp": "tcp",
        "ip": "ip",
        "tokeniz": "token",
        "stop": "stop",
        "stem": "stem",
        "lemma": "lemma",
        "normal": "normal",
        "embedding": "embedding",
    }

    for key, value in mappings.items():
        if key in q:
            return value
    return None



def run_query(question: str):
    logger.info(f"Query: {question}")

    if not Path(INDEX_PATH).exists():
        logger.error(f"FAISS index not found at {INDEX_PATH}")
        return {
            "result": "Index not found. Run ingestion first.",
            "source_documents": [],
            "usage": {},
        }

    logger.info("Initializing Bedrock client...")
    client = _get_bedrock_client()

    logger.info(f"Using embedding model: {BEDROCK_EMBED_MODEL}")
    embedder = TitanEmbedder(BEDROCK_EMBED_MODEL, REGION_NAME)

    logger.info(f"Loading FAISS index from {INDEX_PATH}...")
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embedder.embed_query,  
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded successfully")

    #Section filtering
    all_docs = list(vectorstore.docstore._dict.values())
    logger.info(f"Total documents in vectorstore: {len(all_docs)}")

    section_hint = guess_section(question)
    if section_hint:
        filtered = [
            d for d in all_docs
            if section_hint in d.metadata.get("section", "").lower()
        ]
        if not filtered:
            filtered = all_docs
        logger.info(f"Section filter '{section_hint}': {len(filtered)} docs")
    else:
        filtered = all_docs
        logger.info("No section filter applied")

    #Similarity search
    logger.info("Performing similarity search...")
    query_embedding = embedder.embed_query(question)  
    
    docs_with_scores = vectorstore.similarity_search_with_score_by_vector(
        query_embedding,
        k=RAW_TOP_K
    )
    
    candidate_docs = [doc for doc, _ in docs_with_scores]
    logger.info(f" Found {len(candidate_docs)} candidate documents")

    if not candidate_docs:
        logger.warning("No candidate documents found")
        return {"result": REFUSAL_MESSAGE, "source_documents": [], "usage": {}}

    #Reranking
    logger.info(f"Reranking with {BEDROCK_LLM_MODEL}...")
    scored_docs = rerank_chunks_scored(
        question, candidate_docs, client, BEDROCK_LLM_MODEL
    )
    
    logger.info(f"Scored {len(scored_docs)} documents")
    if scored_docs:
        logger.info(f"Top score: {scored_docs[0][1]}")

    if not scored_docs:
        logger.warning("No documents passed reranking")
        return {"result": REFUSAL_MESSAGE, "source_documents": [], "usage": {}}

    logger.info("Applying diversity filter...")
    diverse_docs = apply_page_diversity(scored_docs)
    final_docs = [doc for doc, _ in diverse_docs[:5]]
    logger.info(f"Selected {len(final_docs)} diverse documents")

    parts = [d.page_content[:MAX_CHUNK_CHARS].strip() for d in final_docs]
    context = "\n\n".join(parts)[:MAX_CONTEXT_CHARS]
    logger.info(f"Context built: {len(context)} characters")

    logger.info("Generating final answer...")
    prompt = f"""
You must answer using ONLY the following context.
If the answer is not present in the context, reply with: {REFUSAL_MESSAGE}

### Context ###
{context}

### Question ###
{question}

### Final Answer ###
"""

    llm_result = _invoke_llm_converse(client, BEDROCK_LLM_MODEL, prompt)

    answer_text = llm_result.get("text", "").strip()

    if not answer_text:
        answer_text = REFUSAL_MESSAGE

    logger.info("Answer generated successfully")

    return {
        "result": answer_text,
        "source_documents": final_docs,
        "usage": llm_result.get("usage", {})
    }

def retrieve_answer(question: str):
    return run_query(question)
