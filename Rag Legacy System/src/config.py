#All configuration is centralized here and can be overridden via .env 
# things like document path, index path, chunk size, AWS region, and the Titan model IDs stored here

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Paths
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", str(BASE_DIR / "documents"))
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Bedrock / AWS
REGION_NAME = os.getenv("AWS_REGION", "us-east-1")

TITAN_EMBEDDING_MODEL = os.getenv(
    "BEDROCK_TITAN_EMBED_MODEL",
    "amazon.titan-embed-text-v1"
)

TITAN_LLM_MODEL = os.getenv(
    "BEDROCK_TITAN_LLM_MODEL",
    "amazon.titan-tg1-large"
)

# Timeouts
CONNECT_TIMEOUT = int(os.getenv("CONNECT_TIMEOUT", "30"))
READ_TIMEOUT = int(os.getenv("READ_TIMEOUT", "60"))
