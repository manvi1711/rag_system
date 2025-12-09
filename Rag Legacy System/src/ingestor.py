#Load all PDFs/DOCX/TXT from the documents folder
#Split them into overlapping chunks and embed them with Titan, storing everything in FAISS.”

import os
import logging
from pathlib import Path

import boto3
from botocore.config import Config

from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from .config import (
    DOCUMENT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    REGION_NAME,
    TITAN_EMBEDDING_MODEL,
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
)

logger = logging.getLogger(__name__)

INDEX_PATH = "faiss_index"

# Compute project root and default docs dir from this file's location
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DOC_DIR = BASE_DIR / "documents"


def load_and_split_documents():
    """
    Loads all PDF, DOCX, and TXT files from a documents folder and splits into chunks.
    It first tries DOCUMENT_PATH from config/.env, then falls back to ./documents
    relative to the project root.
    """
    candidate_paths = []

    # 1) Path from config / .env
    if DOCUMENT_PATH:
        candidate_paths.append(Path(DOCUMENT_PATH))

    # 2) Default path: <project_root>/documents
    candidate_paths.append(DEFAULT_DOC_DIR)

    docs = []
    chosen_path = None

    for p in candidate_paths:
        logger.info(f"Trying documents path: {p}")
        if p.exists() and any(p.glob("**/*")):
            chosen_path = p
            break

    if chosen_path is None:
        logger.warning(
            f"No documents found in any of these paths: "
            f"{', '.join(str(p) for p in candidate_paths)}"
        )
        print(
            f"No documents found. Make sure your PDFs/TXTs are in "
            f"{DEFAULT_DOC_DIR} or set DOCUMENT_PATH correctly in .env."
        )
        return None

    logger.info(f"Using documents path: {chosen_path}")

    # Actually load docs
    for file_path in chosen_path.glob("**/*"):
        if not file_path.is_file():
            continue

        file_ext = file_path.suffix.lower()
        file_path_str = str(file_path)

        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(file_path_str)
                docs.extend(loader.load())
                logger.info(f"Loaded PDF: {file_path.name}")

            elif file_ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path_str)
                docs.extend(loader.load())
                logger.info(f"Loaded Word document: {file_path.name}")

            elif file_ext == ".txt":
                loader = TextLoader(file_path_str, encoding="utf-8")
                docs.extend(loader.load())
                logger.info(f"Loaded text file: {file_path.name}")

            else:
                logger.info(f"Skipping unsupported file: {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to read file {file_path.name}: {e}")
            continue

    if not docs:
        logger.warning(f"Documents path exists but no readable files found in {chosen_path}")
        print(f"Documents path exists but no readable files found in {chosen_path}")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"📌 Finished splitting → {len(docs)} documents into {len(chunks)} chunks")
    return chunks


def create_and_store_embeddings(chunks):
    """
    Creates embeddings using Titan and saves FAISS index to disk.
    """
    logger.info("Creating embeddings and storing in FAISS vector store...")

    try:
        config = Config(
            connect_timeout=CONNECT_TIMEOUT,
            read_timeout=READ_TIMEOUT,
            retries={"max_attempts": 3},
        )

        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=REGION_NAME,
            config=config,
        )

        embedder = BedrockEmbeddings(
            client=bedrock_client,
            model_id=TITAN_EMBEDDING_MODEL,
        )

        # Build FAISS index from chunks
        vectorstore = FAISS.from_documents(chunks, embedder)
        vectorstore.save_local(INDEX_PATH)

        logger.info(f"✔ FAISS index saved successfully → {INDEX_PATH}")
        logger.info(f"✔ Embedded chunks stored: {len(chunks)}")
        return vectorstore

    except Exception as e:
        logger.error(f"🔥 ERROR: Failed to store FAISS index: {e}", exc_info=True)
        return None
