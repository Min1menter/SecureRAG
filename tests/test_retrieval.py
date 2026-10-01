from pathlib import Path

from rag.retrieval import BM25, HybridRetriever, reciprocal_rank_fusion, tokenize
from rag.vectorstore import delete_document
from tests.conftest import OverlapReranker


def test_tokenize_keeps_identifiers_and_parts():
    toks = tokenize("Patch CVE-2025-1234 on TCP/443")
    assert "cve-2025-1234" in toks and "2025" in toks and "tcp/443" in toks and "443" in toks


def test_bm25_prefers_exact_identifier():
    corpus = [tokenize("generic network security text"), tokenize("CVE-2025-1234 router bug"), tokenize("other words here")]
    scores = BM25(corpus).scores(tokenize("CVE-2025-1234"))
    assert scores.argmax() == 1 and scores[0] == 0


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert fused[1] > fused[3] and fused[2] > fused[4]


def test_hybrid_finds_right_document(make_vs, embeddings):
    r = HybridRetriever(make_vs(), embeddings, None, min_dense=0.15, top_k=3)
    res = r.retrieve("How do Kerberos tickets work?")
    assert res.relevant and Path(res.candidates[0].doc.metadata["source"]).name == "kerberos.txt"
    assert res.gate_type == "dense"


def test_exact_identifier_query(make_vs, embeddings):
    r = HybridRetriever(make_vs(), embeddings, None, min_dense=0.15)
    res = r.retrieve("CVE-2025-1234")
    assert Path(res.considered[0].doc.metadata["source"]).name == "cve.txt"
    assert res.considered[0].bm25 > 0


def test_irrelevant_query_is_gated_dense(make_vs, embeddings):
    r = HybridRetriever(make_vs(), embeddings, None, min_dense=0.15)
    res = r.retrieve("banana smoothie recipe")
    assert not res.relevant and res.candidates == []


def test_reranker_gate_and_order(make_vs, embeddings):
    r = HybridRetriever(make_vs(), embeddings, OverlapReranker(), min_rerank=0.05, top_k=2)
    good = r.retrieve("firewall rules block connections")
    assert good.gate_type == "rerank" and good.relevant
    assert Path(good.candidates[0].doc.metadata["source"]).name == "firewall.txt"
    assert len(good.candidates) <= 2 and good.candidates[0].rerank >= good.candidates[-1].rerank
    assert not r.retrieve("banana smoothie recipe").relevant


def test_index_refreshes_after_delete(make_vs, embeddings):
    vs = make_vs()
    r = HybridRetriever(vs, embeddings, None, min_dense=0.15)
    src = next(s for s in {c.doc.metadata["source"] for c in r.retrieve("kerberos").considered} if s.endswith("kerberos.txt"))
    delete_document(vs, src)
    sources = {Path(c.doc.metadata["source"]).name for c in r.retrieve("kerberos tickets").considered}
    assert "kerberos.txt" not in sources


def test_empty_store(make_vs, embeddings):
    res = HybridRetriever(make_vs({}), embeddings, None).retrieve("anything")
    assert not res.relevant and res.gate_type == "none"
