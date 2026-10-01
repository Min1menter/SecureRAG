"""All settings in one place. Override any of them with environment variables or a .env file."""
import os
from pathlib import Path

try:  # optional: loads a .env file if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Paths ---
DOCS_DIR = Path("docs")          # put your PDFs / TXT / MD here
DB_DIR = "chroma_db"             # vector store is saved here
LOG_DIR = Path("logs")
EVAL_DATASET = Path("evaluation/dataset.json")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_MB = _int("MAX_FILE_SIZE_MB", 25)

# --- Embeddings (free, runs locally) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Chunking (changing these needs a re-index: delete chroma_db) ---
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)

# --- Retrieval ---
TOP_K = _int("TOP_K", 6)                        # chunks finally sent to the LLM
CANDIDATES = _int("CANDIDATES", 20)             # taken from EACH retriever (dense, BM25)
RERANK_CANDIDATES = _int("RERANK_CANDIDATES", 20)   # fused candidates passed to the reranker
RRF_K = _int("RRF_K", 60)                       # reciprocal-rank-fusion constant

RERANKER_ENABLED = _bool("RERANKER_ENABLED", True)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Relevance gate: if the best score is below the threshold the bot says "not found".
# These are STARTING POINTS - tune them with the evaluation page / evaluation script.
MIN_RERANK_SCORE = _float("MIN_RERANK_SCORE", 0.05)        # used when the reranker is on (0..1)
MIN_DENSE_SIMILARITY = _float("MIN_DENSE_SIMILARITY", 0.25)  # used without a reranker (cosine)

# --- LLM (free options: "ollama" local, or "groq" free-tier key) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
_DEFAULT_MODEL = "llama-3.3-70b-versatile" if LLM_PROVIDER == "groq" else "llama3.2"
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODEL)

# --- Response length / style ---
MAX_TOKENS = _int("MAX_TOKENS", 1500)
TEMPERATURE = _float("TEMPERATURE", 0.3)
OLLAMA_NUM_CTX = _int("OLLAMA_NUM_CTX", 8192)

# --- Security ---
MAX_QUERY_CHARS = _int("MAX_QUERY_CHARS", 2000)
INJECTION_ACTION = os.getenv("INJECTION_ACTION", "flag")    # "flag" (warn + mark) or "drop" (remove passage)
_REDACT = os.getenv("REDACT_CONTEXT", "auto").strip().lower()  # auto = only when the LLM is a remote API
REDACT_ENABLED = _REDACT in {"1", "true", "yes", "on"} or (_REDACT == "auto" and LLM_PROVIDER != "ollama")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
