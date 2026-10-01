"""Cross-encoder reranker: reads (question, passage) together, so it judges relevance far better
than the embedding model that was only used for fast candidate retrieval."""
import logging
import math
from functools import lru_cache
from typing import List, Optional

from rag.config import RERANKER_ENABLED, RERANKER_MODEL

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # heavy import, only when enabled
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: List[str]) -> List[float]:
        """Relevance in 0..1 for every text (higher = more relevant)."""
        raw = [float(x) for x in self.model.predict([(query, t) for t in texts])]
        if any(x < 0.0 or x > 1.0 for x in raw):   # raw logits -> probabilities
            raw = [1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0))) for x in raw]
        return raw


@lru_cache(maxsize=1)
def get_reranker() -> Optional[CrossEncoderReranker]:
    """Returns the reranker, or None if disabled or the model can't be loaded."""
    if not RERANKER_ENABLED:
        return None
    try:
        return CrossEncoderReranker(RERANKER_MODEL)
    except Exception:
        logger.exception("Could not load reranker %s - continuing without it", RERANKER_MODEL)
        return None
