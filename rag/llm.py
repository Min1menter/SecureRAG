"""Chooses which LLM to use. Add new providers here."""
from rag.config import LLM_MODEL, LLM_PROVIDER, MAX_TOKENS, OLLAMA_NUM_CTX, TEMPERATURE


def get_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq  # needs GROQ_API_KEY
        return ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    from langchain_ollama import ChatOllama  # local, no key
    return ChatOllama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        num_ctx=OLLAMA_NUM_CTX,
    )
