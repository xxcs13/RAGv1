"""
Configuration settings for the RAG system.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in your .env file."

# Model Configuration
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 100

# Vector Database Configuration
DEFAULT_VECTORSTORE_DIR = "chromadb_test"

# Retrieval Configuration
DEFAULT_TOP_K = 30
DEFAULT_TOP_N = 10
DEFAULT_LLM_WEIGHT = 0.7
DEFAULT_BATCH_SIZE = 2

# Telemetry Configuration
enable_telemetry = os.getenv("ENABLE_TELEMETRY", "false").lower() == "true"
if not enable_telemetry:
    os.environ.setdefault("CHROMA_TELEMETRY", "false")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    print("Telemetry disabled for privacy and stability. Set ENABLE_TELEMETRY=true in .env to enable.")

# Logging Configuration
_log = logging.getLogger(__name__)

# Supported File Types
SUPPORTED_EXTENSIONS = ['.pdf', '.pptx', '.ppt', '.xls', '.xlsx']
