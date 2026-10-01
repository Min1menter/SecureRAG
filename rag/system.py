"""Builds the whole system once (used by the web app, the terminal chat and the evaluation script)."""
from rag.graph import build_graph
from rag.reranker import get_reranker
from rag.retrieval import HybridRetriever
from rag.vectorstore import get_embeddings, get_vectorstore, sync_docs_folder


def build_system(sync: bool = True):
    """Returns (vectorstore, retriever, graph)."""
    vs = get_vectorstore()
    if sync:
        sync_docs_folder(vs)
    retriever = HybridRetriever(vs, get_embeddings(), get_reranker())
    return vs, retriever, build_graph(vs, retriever=retriever)
