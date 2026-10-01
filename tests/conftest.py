import re
import uuid
import zlib

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag.vectorstore import index_file


class BagOfWordsEmbeddings(Embeddings):
    """Tiny deterministic embedding model so tests need no downloads."""
    DIM = 1024

    def _vec(self, text):
        v = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker:
    """Fake cross-encoder: fraction of query words found in the passage."""
    def score(self, query, texts):
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [len(q & set(re.findall(r"[a-z0-9]+", t.lower()))) / max(len(q), 1) for t in texts]


FILES = {
    "kerberos.txt": "Kerberos is a network authentication protocol. Kerberos uses tickets issued by a key distribution center. " * 3,
    "firewall.txt": "A firewall filters network traffic using rules. Firewalls block unauthorized connections. " * 3,
    "cve.txt": "Advisory CVE-2025-1234 affects the Acme router firmware. Update to patch the vulnerability. " * 3,
}


@pytest.fixture
def embeddings():
    return BagOfWordsEmbeddings()


@pytest.fixture
def make_vs(tmp_path, embeddings):
    def _make(files=None):
        vs = Chroma(
            collection_name="t" + uuid.uuid4().hex[:12],
            persist_directory=str(tmp_path / "db"),
            embedding_function=embeddings,
        )
        docs = tmp_path / "docs"
        docs.mkdir(exist_ok=True)
        for name, text in (FILES if files is None else files).items():
            p = docs / name
            p.write_text(text, encoding="utf-8")
            index_file(vs, p)
        return vs
    return _make
