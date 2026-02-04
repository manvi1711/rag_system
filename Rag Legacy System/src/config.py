# Loads paths, model IDs, chunk settings from .env and provides defaults.

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

#PATHS 
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", str(BASE_DIR / "documents"))
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")

#CHUNKING CONFIG 
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

#AWS CONFIG 
REGION_NAME = os.getenv("AWS_REGION", "us-east-1")

#BEDROCK MODELS

# Embedding model 
BEDROCK_EMBED_MODEL = os.getenv(
    "BEDROCK_TITAN_EMBED_MODEL",
    "amazon.titan-embed-text-v2:0"      
)

# LLM model 
BEDROCK_LLM_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


# TIMEOUTS
CONNECT_TIMEOUT = int(os.getenv("CONNECT_TIMEOUT", "30"))
READ_TIMEOUT = int(os.getenv("READ_TIMEOUT", "60"))
