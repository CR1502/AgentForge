"""Retrieval-Augmented Generation: real Chroma vector store over the corpus.

CONTRACT (consumed by graph.py and scripts/ingest.py):
    - build_vectorstore(reset: bool = False) -> Chroma
        Load + chunk every file in config.CORPUS_DIR, embed with
        OllamaEmbeddings(config.EMBED_MODEL), and persist a Chroma collection
        at config.CHROMA_DIR (collection = config.COLLECTION_NAME).
        If `reset`, delete any existing collection first. Returns the store.
    - get_retriever(k: int = 4)
        Open the persisted Chroma collection and return a retriever.
        Must raise a clear error if the store has not been built yet.
    - retrieve(query: str, k: int = 4) -> str
        Convenience: return the concatenated page_content of the top-k docs,
        formatted for use as grounding context. Empty string if nothing found.

Implemented by the RAG agent in Phase 1.
"""

from __future__ import annotations

# TODO(Phase 1 - RAG agent): implement build_vectorstore / get_retriever /
# retrieve using langchain_chroma.Chroma + langchain_ollama.OllamaEmbeddings.
