"""Turns a retrieved chunk into a human-readable citation."""
from pathlib import Path

from langchain_core.documents import Document


def format_citation(doc: Document) -> str:
    name = Path(doc.metadata.get("source", "unknown")).name
    page = doc.metadata.get("page")            # PyPDFLoader pages are 0-based
    return f"{name} — Page {page + 1}" if isinstance(page, int) else name
