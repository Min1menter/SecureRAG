"""Step 2 of RAG: embeddings + Chroma. Index, update, list and delete documents."""
import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

from langchain_chroma import Chroma

from rag.config import DB_DIR, DOCS_DIR, EMBED_MODEL, SUPPORTED_EXTENSIONS
from rag.ingestion import file_hash, load_and_split

logger = logging.getLogger(__name__)

_corpus_version = 0     # bumped on every change so retrievers know to rebuild their index


def corpus_version() -> int:
    return _corpus_version


def _bump() -> None:
    global _corpus_version
    _corpus_version += 1


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings  # imported lazily (heavy)
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())


def index_file(vs: Chroma, path: Path) -> Tuple[str, int]:
    """Index one file. Returns (status, chunks): 'added', 'updated' or 'skipped'.

    Uses a content hash, so a file with the same name but new content is re-indexed.
    """
    source = str(path)
    new_hash = file_hash(path)
    existing = vs.get(where={"source": source}, include=["metadatas"])

    if existing["ids"] and existing["metadatas"][0].get("file_hash") == new_hash:
        return "skipped", 0

    chunks = load_and_split(path)          # load first: a bad new version keeps the old vectors
    if not chunks:
        raise ValueError("No readable text found (is it a scanned PDF?).")

    status = "added"
    if existing["ids"]:
        vs.delete(ids=existing["ids"])
        status = "updated"
    vs.add_documents(chunks)
    _bump()
    logger.info("%s %s (%d chunks)", status, source, len(chunks))
    return status, len(chunks)


def delete_document(vs: Chroma, source: str) -> int:
    """Remove a document's vectors and its file. Returns number of chunks deleted."""
    ids = vs.get(where={"source": source})["ids"]
    if ids:
        vs.delete(ids=ids)
    Path(source).unlink(missing_ok=True)
    _bump()
    logger.info("deleted %s (%d chunks)", source, len(ids))
    return len(ids)


def list_documents(vs: Chroma) -> Dict[str, int]:
    """{source path: number of chunks}"""
    metas = vs.get(include=["metadatas"])["metadatas"]
    return dict(sorted(Counter((m or {}).get("source", "?") for m in metas).items()))


def sync_docs_folder(vs: Chroma) -> int:
    """Index every new or changed supported file in DOCS_DIR. Returns chunks written."""
    DOCS_DIR.mkdir(exist_ok=True)
    total = 0
    for f in sorted(DOCS_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                total += index_file(vs, f)[1]
            except Exception:
                logger.exception("Failed to index %s", f)
    return total


def is_empty(vs: Chroma) -> bool:
    return not vs.get(limit=1)["ids"]
