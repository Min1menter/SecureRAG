"""Writes logs to logs/rag.log (safe to call many times, e.g. on Streamlit reruns)."""
import logging

from rag.config import LOG_DIR, LOG_LEVEL


def setup_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_rag_configured", False):
        return
    LOG_DIR.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / "rag.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)
    root._rag_configured = True
