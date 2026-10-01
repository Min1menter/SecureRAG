"""Step 1 of RAG: validate a file, read it, cut it into chunks."""
import hashlib
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS


def validate_upload(filename: str, size_bytes: int, head: bytes = b"") -> str:
    """Return a safe file name or raise ValueError with a user-friendly message."""
    safe = Path(filename).name                      # strips any folder parts
    if not safe or safe.startswith("."):
        raise ValueError("Invalid file name.")
    suffix = Path(safe).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Use PDF, TXT or MD.")
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File too large (limit {MAX_FILE_SIZE_MB} MB).")
    if suffix == ".pdf" and head and not head.startswith(b"%PDF"):
        raise ValueError("This doesn't look like a real PDF file.")
    return safe


def file_hash(path: Path) -> str:
    """SHA-256 of the file contents, used to detect changed files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_and_split(path: Path) -> List[Document]:
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(loader.load())

    digest = file_hash(path)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({"file_hash": digest, "chunk_id": f"{path.stem}-{i}"})
    return chunks
