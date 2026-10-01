"""Hybrid retrieval.

question -> dense search (cosine) + keyword search (BM25) -> reciprocal-rank fusion
         -> optional cross-encoder rerank -> relevance gate ("is anything actually relevant?")

The index is held in memory (embeddings are read back from Chroma), which is simple, exact and
fast for up to tens of thousands of chunks. Chroma remains the persistent store.
"""
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from langchain_core.documents import Document

from rag.citations import format_citation
from rag.config import (
    CANDIDATES, MIN_DENSE_SIMILARITY, MIN_RERANK_SCORE, RERANK_CANDIDATES, RRF_K, TOP_K,
)
from rag.vectorstore import corpus_version

logger = logging.getLogger(__name__)

# ---------- BM25 keyword search ----------
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)*")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are", "was", "were", "be",
    "what", "which", "how", "does", "do", "did", "it", "its", "this", "that", "with", "as", "by",
    "at", "from", "about", "me", "tell", "explain", "can", "you",
}


def tokenize(text: str) -> List[str]:
    """Lowercase tokens. Identifiers like CVE-2025-1234 or TCP/443 are kept whole AND split into parts."""
    tokens: List[str] = []
    for m in _TOKEN.findall(text.lower()):
        if m in _STOPWORDS:
            continue
        tokens.append(m)
        if re.search(r"[-_/.]", m):
            tokens.extend(p for p in re.split(r"[-_/.]", m) if p and p not in _STOPWORDS)
    return tokens


class BM25:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.n = len(corpus_tokens)
        self.k1, self.b = k1, b
        self.doc_len = np.array([len(t) for t in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 1.0
        self.postings: Dict[str, List[tuple]] = defaultdict(list)   # token -> [(doc index, tf)]
        for i, toks in enumerate(corpus_tokens):
            for tok, tf in Counter(toks).items():
                self.postings[tok].append((i, tf))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for tok in set(query_tokens):
            for i, tf in self.postings.get(tok, ()):
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s[i] += self.idf[tok] * tf * (self.k1 + 1) / norm
        return s


# ---------- Corpus snapshot ----------
class CorpusIndex:
    def __init__(self, docs: List[Document], emb: Optional[np.ndarray]):
        self.docs = docs
        self.size = len(docs)
        if self.size:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self.emb = emb / np.maximum(norms, 1e-12)
        else:
            self.emb = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25([tokenize(d.page_content) for d in docs])


def load_corpus(vs) -> CorpusIndex:
    data = vs.get(include=["documents", "metadatas", "embeddings"])
    emb = data.get("embeddings")
    if emb is None or len(emb) == 0:
        return CorpusIndex([], None)
    docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(data["documents"], data["metadatas"])]
    return CorpusIndex(docs, np.asarray(emb, dtype=np.float32))


# ---------- Results ----------
@dataclass
class Candidate:
    doc: Document
    dense: float                    # cosine similarity to the query
    bm25: float                     # keyword score
    fused: float                    # reciprocal-rank-fusion score
    rerank: Optional[float] = None  # cross-encoder relevance 0..1

    def to_dict(self, rank: int) -> dict:
        page = self.doc.metadata.get("page")
        return {
            "rank": rank,
            "citation": format_citation(self.doc),
            "source": Path(self.doc.metadata.get("source", "?")).name,
            "page": page + 1 if isinstance(page, int) else None,
            "dense": round(self.dense, 3),
            "bm25": round(self.bm25, 2),
            "fused": round(self.fused, 4),
            "rerank": None if self.rerank is None else round(self.rerank, 3),
            "text": self.doc.page_content[:600],
        }


@dataclass
class RetrievalResult:
    query: str
    candidates: List[Candidate]            # chunks that passed the gate (what the LLM will see)
    considered: List[Candidate]            # everything that was scored, best first
    relevant: bool
    gate_score: float
    gate_type: str                         # "rerank" | "dense" | "none"
    threshold: float
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "relevant": self.relevant,
            "gate_score": round(self.gate_score, 3),
            "gate_type": self.gate_type,
            "threshold": self.threshold,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "candidates": [c.to_dict(i) for i, c in enumerate(self.considered[:10], 1)],
        }


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = RRF_K) -> Dict[int, float]:
    fused: Dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            fused[idx] += 1.0 / (k + rank)
    return dict(fused)


# ---------- Retriever ----------
class HybridRetriever:
    def __init__(
        self, vectorstore, embeddings, reranker=None, *,
        top_k: int = TOP_K, candidates: int = CANDIDATES, rerank_candidates: int = RERANK_CANDIDATES,
        min_dense: float = MIN_DENSE_SIMILARITY, min_rerank: float = MIN_RERANK_SCORE,
    ):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.reranker = reranker
        self.top_k, self.candidates, self.rerank_candidates = top_k, candidates, rerank_candidates
        self.min_dense, self.min_rerank = min_dense, min_rerank
        self._index: Optional[CorpusIndex] = None
        self._index_version = -1

    def _get_index(self) -> CorpusIndex:
        version = corpus_version()
        if self._index is None or version != self._index_version:
            self._index = load_corpus(self.vectorstore)
            self._index_version = version
            logger.info("retrieval index rebuilt: %d chunks", self._index.size)
        return self._index

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        idx = self._get_index()
        if idx.size == 0:
            return RetrievalResult(query, [], [], False, 0.0, "none", 0.0, (time.perf_counter() - t0) * 1000)

        qv = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
        dense = idx.emb @ qv
        bm25 = idx.bm25.scores(tokenize(query))

        dense_rank = [int(i) for i in np.argsort(-dense)[: self.candidates]]
        bm25_rank = [int(i) for i in np.argsort(-bm25)[: self.candidates] if bm25[i] > 0]
        fused = reciprocal_rank_fusion([dense_rank, bm25_rank])
        top = sorted(fused, key=fused.get, reverse=True)[: self.rerank_candidates]
        cands = [Candidate(idx.docs[i], float(dense[i]), float(bm25[i]), fused[i]) for i in top]

        kept: Optional[List[Candidate]] = None
        gate_type, threshold, gate_score = "dense", self.min_dense, 0.0

        if self.reranker is not None:
            try:
                scores = self.reranker.score(query, [c.doc.page_content for c in cands])
                for c, s in zip(cands, scores):
                    c.rerank = float(s)
                cands.sort(key=lambda c: c.rerank, reverse=True)
                gate_type, threshold, gate_score = "rerank", self.min_rerank, cands[0].rerank
                kept = [c for c in cands if c.rerank >= threshold][: self.top_k]
            except Exception:
                logger.exception("Reranking failed; falling back to dense relevance")

        if kept is None:                                   # no reranker (or it failed)
            gate_score = max(c.dense for c in cands)
            kept = cands[: self.top_k] if gate_score >= threshold else []

        return RetrievalResult(
            query=query, candidates=kept, considered=cands, relevant=bool(kept),
            gate_score=gate_score, gate_type=gate_type, threshold=threshold,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
