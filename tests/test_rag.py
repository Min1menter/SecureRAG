"""Run from the project folder:  python -m pytest -q"""
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.citations import format_citation
from rag.ingestion import validate_upload
from rag.vectorstore import delete_document, index_file, list_documents


@pytest.fixture
def vs(tmp_path):
    return Chroma(
        collection_name="test",
        persist_directory=str(tmp_path / "db"),
        embedding_function=DeterministicFakeEmbedding(size=32),
    )


def test_validate_upload():
    assert validate_upload("../../etc/notes.txt", 100) == "notes.txt"
    with pytest.raises(ValueError):
        validate_upload("virus.exe", 100)
    with pytest.raises(ValueError):
        validate_upload("big.pdf", 999 * 1024 * 1024)
    with pytest.raises(ValueError):
        validate_upload("fake.pdf", 100, b"MZ\x90\x00")
    with pytest.raises(ValueError):
        validate_upload(".hidden.txt", 100)


def test_citation_pages_are_one_based():
    assert format_citation(Document(page_content="x", metadata={"source": "docs/a.pdf", "page": 16})) == "a.pdf — Page 17"
    assert format_citation(Document(page_content="x", metadata={"source": "docs/a.md"})) == "a.md"


def test_index_skip_update_delete(vs, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Firewalls filter traffic. " * 100, encoding="utf-8")

    status, n = index_file(vs, f)
    assert status == "added" and n > 0
    assert index_file(vs, f) == ("skipped", 0)

    f.write_text("Zero trust verifies every request. " * 100, encoding="utf-8")
    status, _ = index_file(vs, f)
    assert status == "updated"
    assert list_documents(vs) == {str(f): list_documents(vs)[str(f)]}
    assert "Zero trust" in vs.get(include=["documents"])["documents"][0]

    assert delete_document(vs, str(f)) > 0
    assert list_documents(vs) == {}
    assert not Path(f).exists()
