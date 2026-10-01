from rag.evaluation import keyword_coverage, recall_at_k, reciprocal_rank, run_evaluation
from rag.retrieval import HybridRetriever


def test_metric_functions():
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0
    assert keyword_coverage("Kerberos uses Tickets", ["tickets", "kdc"]) == 0.5


def test_run_evaluation_retrieval_only(make_vs, embeddings):
    r = HybridRetriever(make_vs(), embeddings, None, min_dense=0.15)
    dataset = [
        {"question": "How do Kerberos tickets work?", "expected_sources": ["kerberos.txt"]},
        {"question": "What does a firewall do with traffic rules?", "expected_sources": ["firewall.txt"]},
        {"question": "banana smoothie recipe", "answerable": False},
        {"question": "Missing file question", "expected_sources": ["ghost.pdf"]},
    ]
    report = run_evaluation(r, dataset, indexed_sources={"kerberos.txt", "firewall.txt", "cve.txt"})
    s = report["summary"]
    assert s["abstention_accuracy_on_unanswerable"] == 1.0
    assert s["recall@5"] == round(2 / 3, 3)        # the ghost.pdf question can never be found
    assert any("ghost.pdf" in w for w in report["warnings"])
