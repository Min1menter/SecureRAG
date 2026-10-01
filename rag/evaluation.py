"""Measure the RAG system instead of eyeballing it.

Dataset format (evaluation/dataset.json):
  {"questions": [
     {"question": "...", "expected_sources": ["file.pdf"], "expected_keywords": ["a", "b"]},
     {"question": "Something NOT in your docs", "answerable": false}
  ]}
"""

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# ---------- Pure metric functions ----------
def recall_at_k(ranked_sources: List[str], expected: List[str], k: int) -> float:
    return 1.0 if any(s in expected for s in ranked_sources[:k]) else 0.0


def reciprocal_rank(ranked_sources: List[str], expected: List[str]) -> float:
    for i, s in enumerate(ranked_sources, 1):
        if s in expected:
            return 1.0 / i
    return 0.0


def keyword_coverage(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    low = answer.lower()
    return sum(k.lower() in low for k in keywords) / len(keywords)


def _unique(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def load_dataset(path) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


JUDGE_PROMPT = """You are grading a RAG answer. Using ONLY the context, rate from 1 to 5 how well \
the answer is supported by the context (5 = every claim is supported, 1 = mostly unsupported or invented).
Reply with just the number.

Context:
{context}

Question: {question}

Answer: {answer}

Score:"""


def judge_faithfulness(llm, question: str, context_docs, answer: str) -> Optional[int]:
    """LLM-as-judge (noisy; use it to compare versions, not as an absolute truth)."""
    context = "\n\n".join(d.page_content[:800] for d in context_docs) or "(none)"
    try:
        out = llm.invoke(JUDGE_PROMPT.format(context=context, question=question, answer=answer))
        m = re.search(r"[1-5]", str(getattr(out, "content", out)))
        return int(m.group(0)) if m else None
    except Exception:
        return None


# ---------- Runner ----------
def run_evaluation(
    retriever, dataset: List[dict], graph=None, judge_llm=None,
    ks: Sequence[int] = (1, 3, 5), indexed_sources: Optional[set] = None,
) -> Dict:
    rows, warnings = [], []
    for q in dataset:
        question = q["question"]
        answerable = q.get("answerable", True)
        expected = q.get("expected_sources", [])
        keywords = q.get("expected_keywords", [])

        if indexed_sources is not None:
            missing = [s for s in expected if s not in indexed_sources]
            if missing:
                warnings.append(f"'{question[:50]}': expected file(s) not indexed: {missing}")

        result = retriever.retrieve(question)
        ranked = _unique(Path(c.doc.metadata.get("source", "?")).name for c in result.considered)

        row = {
            "question": question,
            "answerable": answerable,
            "top_source": ranked[0] if ranked else None,
            "gate_passed": result.relevant,
            "gate_score": round(result.gate_score, 3),
            "retrieval_ms": round(result.elapsed_ms, 1),
        }
        if answerable and expected:
            for k in ks:
                row[f"recall@{k}"] = recall_at_k(ranked, expected, k)
            row["rr"] = reciprocal_rank(ranked, expected)

        if graph is not None:
            cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
            out = graph.invoke({"messages": [("user", question)]}, cfg)
            answer = out["messages"][-1].content
            row["answer"] = answer
            row["generation_ms"] = round(out.get("generation_ms") or 0.0, 1)
            row["has_citation"] = bool(re.search(r"\[\d+\]", answer))
            if answerable and keywords:
                row["keyword_coverage"] = round(keyword_coverage(answer, keywords), 3)
            if judge_llm is not None and answerable:
                row["faithfulness"] = judge_faithfulness(judge_llm, question, out.get("context", []), answer)
        rows.append(row)

    scored = [r for r in rows if "rr" in r]
    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]
    summary = {
        "questions": len(rows),
        "answerable": len(answerable_rows),
        "mrr": _mean([r["rr"] for r in scored]),
        "answer_rate_on_answerable": _mean([float(r["gate_passed"]) for r in answerable_rows]),
        "abstention_accuracy_on_unanswerable": _mean([float(not r["gate_passed"]) for r in unanswerable_rows]),
        "avg_retrieval_ms": _mean([r["retrieval_ms"] for r in rows]),
    }
    for k in ks:
        summary[f"recall@{k}"] = _mean([r[f"recall@{k}"] for r in scored])
    if graph is not None:
        summary["avg_keyword_coverage"] = _mean([r["keyword_coverage"] for r in rows if "keyword_coverage" in r])
        summary["citation_rate"] = _mean([float(r["has_citation"]) for r in answerable_rows if "has_citation" in r])
        summary["avg_generation_ms"] = _mean([r["generation_ms"] for r in rows if "generation_ms" in r])
        if judge_llm is not None:
            summary["avg_faithfulness_1to5"] = _mean([r["faithfulness"] for r in rows if r.get("faithfulness")])
    return {"summary": summary, "rows": rows, "warnings": warnings}