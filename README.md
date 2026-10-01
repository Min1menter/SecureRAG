# SecureRAG — conversational, hybrid-retrieval RAG chatbot (LangChain + LangGraph, free models)

Chat with your own PDF / TXT / MD files. Free stack: local embeddings + local reranker + Ollama (local) or Groq (free tier).

## How a question flows
```
question -> validate -> rewrite (follow-ups become standalone)
         -> dense search (cosine) + BM25 keyword search -> reciprocal-rank fusion
         -> cross-encoder rerank -> relevance gate
              |-- nothing relevant -> "I couldn't find this in your documents" (LLM not called)
              '-- relevant -> injection check + secret redaction -> LLM -> output validation -> answer + [1][2] citations
```

## Features
- **Streamlit UI** with 4 pages: Chat · Documents · Retrieval Debug · Evaluation (+ chat export)
- **Hybrid retrieval**: semantic + BM25 (exact IDs like `CVE-2025-1234`, `TCP/443` work), fused with RRF
- **Reranking** with a cross-encoder, and a **relevance gate** that refuses to answer from weak matches
- **Conversational query rewriting** ("what about its weaknesses?" → standalone question)
- **Citations** with page numbers: `[1] security.pdf — Page 17`
- **Security layer**: untrusted-context prompt, prompt-injection detection (flag or drop), secret/PII redaction before text goes to a remote LLM, prompt-leak canary, citation validation, upload validation
- **Document management**: content-hash re-indexing, replace, delete
- **Evaluation**: Recall@K, MRR, abstention accuracy, keyword coverage, citation rate, latency, optional LLM-judged faithfulness
- Logging (`logs/rag.log`), env-configurable settings, 33 unit tests

## Structure
```
rag_chatbot/
├── app.py  cli.py  requirements.txt  .env.example
├── docs/                 # your documents
├── evaluation/           # dataset.json + evaluate.py (CLI)
├── tests/
└── rag/
    ├── config.py         # all settings (env-overridable)
    ├── ingestion.py      # validation, hashing, chunking
    ├── vectorstore.py    # embeddings + Chroma (index/update/delete/list)
    ├── retrieval.py      # BM25 + dense + RRF + rerank + relevance gate
    ├── reranker.py       # cross-encoder
    ├── query.py          # follow-up question rewriting
    ├── security.py       # injection detection, redaction, output validation
    ├── graph.py          # LangGraph pipeline
    ├── citations.py  llm.py  system.py  evaluation.py  logging_config.py
```
Keep the `rag/` folder next to `app.py` when copying files.

## Run
```bash
pip install -r requirements.txt
ollama pull llama3.2                 # or use Groq (below)
streamlit run app.py                 # web UI
python cli.py                        # terminal chat
python -m pytest -q                  # tests (no model downloads needed)
python -m evaluation.evaluate        # retrieval metrics on evaluation/dataset.json
```
First run downloads the embedding model and the reranker (~100 MB total). Groq: set `GROQ_API_KEY` and `LLM_PROVIDER=groq` (see `.env.example`).

## Evaluate it (do this before trusting the thresholds)
1. Edit `evaluation/dataset.json`: questions about *your* docs, `expected_sources` = file names, `expected_keywords` = words a good answer contains, and a few `"answerable": false` questions your docs do NOT cover.
2. Run `python -m evaluation.evaluate` (add `--generate` for answers, `--judge` for faithfulness) or use the Evaluation page.
3. Use the **Retrieval Debug** page to see scores, then tune `MIN_RERANK_SCORE` / `MIN_DENSE_SIMILARITY`, `TOP_K`, `CHUNK_SIZE` in `.env`.
   Too many "not found" on answerable questions → lower the threshold. Answers to unanswerable questions → raise it.

## Known limits (honest list)
- Relevance thresholds are starting points; tune them on your own documents.
- The retrieval index is held in memory (fine for tens of thousands of chunks; beyond that use Chroma queries + a BM25 service).
- Injection detection is pattern-based and can be evaded; it is one layer, not a guarantee.
- Chunking is character-based, not structure-aware (sections/tables are not respected).
- Scanned PDFs (no text layer) are rejected; there is no OCR.
- Faithfulness via LLM-judge is noisy; use it to compare versions, not as absolute truth.
