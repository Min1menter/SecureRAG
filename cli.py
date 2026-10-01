"""Terminal chat.   python cli.py   (add --reindex to rebuild the index)"""
import shutil
import sys
from pathlib import Path

from rag.citations import format_citation
from rag.config import DB_DIR, DOCS_DIR
from rag.logging_config import setup_logging
from rag.security import validate_query
from rag.system import build_system
from rag.vectorstore import is_empty


def main():
    setup_logging()
    if "--reindex" in sys.argv and Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR)

    vs, _retriever, graph = build_system()
    if is_empty(vs):
        sys.exit(f"No documents found. Add PDF/TXT/MD files to ./{DOCS_DIR} and rerun.")

    config = {"configurable": {"thread_id": "cli-session"}}
    print("RAG chatbot ready. Type 'exit' to quit.\n")

    while True:
        raw = input("You: ").strip()
        if raw.lower() in {"exit", "quit"}:
            break
        try:
            q = validate_query(raw)
        except ValueError as e:
            print(f"  ! {e}\n")
            continue

        result = graph.invoke({"messages": [("user", q)]}, config)
        searched = result.get("standalone_question", q)
        if searched != q:
            print(f"\n(searched for: {searched})")
        print(f"\nBot: {result['messages'][-1].content}\n")
        for i, d in enumerate(result.get("context", []), 1):
            print(f"  [{i}] {format_citation(d)}")
        for flag in result.get("security_flags", []):
            print(f"  [security] {flag}")
        info = result.get("retrieval", {})
        if info:
            print(f"  (retrieval {info['elapsed_ms']:.0f} ms, {info['gate_type']} score {info['gate_score']})")
        print()


if __name__ == "__main__":
    main()
