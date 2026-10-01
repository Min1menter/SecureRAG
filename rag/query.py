"""Conversational query rewriting: turns "what about its weaknesses?" into a standalone search question."""
import logging
from typing import List

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 6      # last 3 turns
MAX_HISTORY_CHARS = 400       # per message, keeps the rewrite prompt small
MAX_REWRITE_CHARS = 500

REWRITE_PROMPT = """Given the chat history and the user's latest message, rewrite the latest \
message as ONE standalone question that can be understood without the history.
Replace pronouns and vague references (it, its, that, the second one) with what they refer to.
Do not answer the question. If it is already standalone, return it unchanged.
Output only the question.

Chat history:
{history}

Latest message: {question}

Standalone question:"""


def rewrite_question(llm, messages: List) -> str:
    """Return a standalone version of the last message. Falls back to the original on any problem."""
    question = messages[-1].content
    prior = messages[:-1][-MAX_HISTORY_MESSAGES:]
    if not prior:                          # first turn: nothing to resolve, skip the LLM call
        return question

    history = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {str(m.content)[:MAX_HISTORY_CHARS]}"
        for m in prior
    )
    try:
        out = llm.invoke(REWRITE_PROMPT.format(history=history, question=question))
        text = str(getattr(out, "content", out)).strip()
    except Exception:
        logger.exception("Query rewrite failed; using the original question")
        return question

    rewritten = text.splitlines()[0].strip().strip('"') if text else ""
    if not rewritten or len(rewritten) > MAX_REWRITE_CHARS:
        return question
    logger.info("rewrote %r -> %r", question, rewritten)
    return rewritten
