"""Security layer for RAG.

Threat model: uploaded documents are UNTRUSTED. They may contain instructions aimed at the LLM
(indirect prompt injection) or secrets that should not be sent to a remote LLM API.

This is defence in depth, not a guarantee: regex detection is easy to evade, so the main
protection is still the prompt (context = data, never instructions) plus output checks.
"""
import re
import secrets
from typing import List, Optional, Tuple

from rag.config import MAX_QUERY_CHARS

# A random marker placed in the system prompt. If it ever shows up in an answer, the prompt leaked.
CANARY = f"CANARY-{secrets.token_hex(6)}"
REFUSAL = "I can't share that."

# ---------- Input validation ----------
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def validate_query(text: str) -> str:
    """Clean a user question or raise ValueError with a friendly message."""
    cleaned = _CONTROL.sub("", text or "").strip()
    if not cleaned:
        raise ValueError("Please type a question.")
    if len(cleaned) > MAX_QUERY_CHARS:
        raise ValueError(f"Question too long (limit {MAX_QUERY_CHARS} characters).")
    return cleaned


# ---------- Prompt-injection detection ----------
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
    r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|system)",
    r"(?:reveal|show|print|repeat|output|leak)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system|hidden|initial)\s+(?:prompt|instructions?)",
    r"you\s+are\s+now\s+(?:a|an|in|the)\b",
    r"(?:enter|enable|switch\s+to)\s+(?:developer|debug|god)\s+mode",
    r"new\s+instructions?\s*:",
    r"override\s+(?:the\s+)?(?:system|safety)\s+(?:prompt|instructions?|rules?)",
    r"<\s*/?\s*(?:system|assistant)\s*>",
    r"do\s+anything\s+now",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


def find_injection(text: str) -> Optional[str]:
    """Return the suspicious snippet if the text contains instruction-like content, else None."""
    m = _INJECTION.search(text or "")
    return m.group(0)[:80] if m else None


# ---------- Secret / PII redaction ----------
_SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL), "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b(\s*[:=]\s*)(['\"]?)[^\s'\"]{6,}"), r"\1\2\3[REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
]


def redact_sensitive(text: str) -> Tuple[str, int]:
    """Mask API keys, tokens, private keys, passwords and e-mail addresses. Returns (text, count)."""
    total = 0
    for pattern, repl in _SECRET_PATTERNS:
        text, n = pattern.subn(repl, text)
        total += n
    return text, total


# ---------- Output validation ----------
_CITATION = re.compile(r"\[(\d{1,3})\]")


def validate_output(answer: str, n_sources: int, redact: bool = True) -> Tuple[str, List[str]]:
    """Check an LLM answer. Returns (clean_answer, flags describing what was changed)."""
    if CANARY in answer:
        return REFUSAL, ["Blocked an answer that leaked the system prompt."]

    flags: List[str] = []
    if redact:
        answer, n = redact_sensitive(answer)
        if n:
            flags.append(f"Redacted {n} sensitive value(s) from the answer.")

    removed = []

    def _fix(m: "re.Match") -> str:
        num = int(m.group(1))
        if 1 <= num <= n_sources:
            return m.group(0)
        removed.append(num)
        return ""

    answer = _CITATION.sub(_fix, answer)
    if removed:
        flags.append(f"Removed citation(s) to non-existent sources: {sorted(set(removed))}.")
    return answer, flags
