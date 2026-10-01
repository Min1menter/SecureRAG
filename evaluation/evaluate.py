"""Run the evaluation from the terminal (from the project folder):

    python -m evaluation.evaluate                       # retrieval metrics only (fast)
    python -m evaluation.evaluate --generate            # also generate answers
    python -m evaluation.evaluate --generate --judge    # + LLM-judged faithfulness
"""
import argparse
import json
from pathlib import Path

from rag.config import EVAL_DATASET
from rag.evaluation import load_dataset, run_evaluation
from rag.llm import get_llm
from rag.logging_config import setup_logging
from rag.system import build_system
from rag.vectorstore import list_documents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(EVAL_DATASET))
    ap.add_argument("--generate", action="store_true", help="also generate answers (slower)")
    ap.add_argument("--judge", action="store_true", help="LLM-judged faithfulness (needs --generate)")
    ap.add_argument("--out", default="evaluation/results.json")
    args = ap.parse_args()

    setup_logging()
    vs, retriever, graph = build_system()
    indexed = {Path(s).name for s in list_documents(vs)}
    report = run_evaluation(
        retriever, load_dataset(args.dataset),
        graph=graph if args.generate else None,
        judge_llm=get_llm() if (args.generate and args.judge) else None,
        indexed_sources=indexed,
    )

    for w in report["warnings"]:
        print("WARNING:", w)
    print("\nRAG evaluation")
    print("-" * 40)
    for k, v in report["summary"].items():
        print(f"{k:38s} {v}")
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
