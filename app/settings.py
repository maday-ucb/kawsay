"""Paths and fixed parameters for the Kawsay retrieval system."""
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"
DOCUMENT_DIR = DATA_DIR / "documents"
RAW_PDF_DIR  = DATA_DIR / "raw_pdfs"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR    = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"

# Embedding model (multilingual, 512-token limit, works for Spanish and indigenous language texts)
EMBEDDING_ID = "intfloat/multilingual-e5-base"

# Ollama generator
OLLAMA_MODEL   = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval
TOP_K             = 5
MAX_QUERY_LENGTH  = 500

# Chunking: multilingual-e5-base limit is 512 tokens ≈ 380 words; 400 words stays safely under
CHUNK_MAX_WORDS = 400
CHUNK_OVERLAP   = 60

# Gate: minimum semantic score to attempt generation
DEFAULT_THRESHOLD = 0.30
