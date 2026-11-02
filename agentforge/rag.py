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

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentforge import config

# Chunking parameters shared by ingest + any rebuild.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Separator joining retrieved chunks in retrieve().
_CHUNK_SEPARATOR = "\n\n---\n\n"


def _embeddings() -> OllamaEmbeddings:
    """Construct the Ollama embedding client from config."""
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def _load_corpus_documents() -> list[Document]:
    """Read every ``*.md`` file under config.CORPUS_DIR into Documents.

    Each Document carries ``metadata={"source": <filename>}`` so retrieved
    chunks can be attributed back to their origin file.
    """
    corpus_dir = config.CORPUS_DIR
    if not corpus_dir.exists():
        raise RuntimeError(
            f"Corpus directory does not exist: {corpus_dir}. "
            "Create markdown files under data/corpus/ before ingesting."
        )

    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        docs.append(Document(page_content=text, metadata={"source": path.name}))

    if not docs:
        raise RuntimeError(
            f"No non-empty *.md files found in {corpus_dir}. "
            "Add corpus documents before ingesting."
        )
    return docs


def build_vectorstore(reset: bool = False) -> Chroma:
    """Load + chunk the corpus, embed it, and persist a Chroma collection.

    Args:
        reset: If True, delete any existing persisted store first so the
            rebuild is clean (no stale chunks from removed files).

    Returns:
        The populated, persisted Chroma vector store.
    """
    persist_dir = config.CHROMA_DIR

    if reset and persist_dir.exists():
        # Wipe the whole persist directory for a guaranteed-clean rebuild.
        shutil.rmtree(persist_dir)

    documents = _load_corpus_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    persist_dir.mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=str(persist_dir),
    )

    if reset:
        # Even if the directory was absent, ensure no stale collection lingers.
        try:
            store.delete_collection()
        except Exception:
            pass
        store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=_embeddings(),
            persist_directory=str(persist_dir),
        )

    store.add_documents(chunks)
    return store


def _open_store() -> Chroma:
    """Open the persisted Chroma collection, erroring if it was never built."""
    persist_dir = config.CHROMA_DIR
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        raise RuntimeError(
            f"Vector store not found at {persist_dir}. "
            "Run `python scripts/ingest.py --reset` to build it first."
        )

    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=str(persist_dir),
    )

    # A persist dir can exist but hold no vectors (e.g. an interrupted build).
    try:
        count = store._collection.count()
    except Exception:
        count = None
    if count == 0:
        raise RuntimeError(
            f"Vector store at {persist_dir} is empty. "
            "Run `python scripts/ingest.py --reset` to (re)build it."
        )

    return store


def get_retriever(k: int = 4):
    """Return a retriever over the persisted Chroma collection.

    Raises:
        RuntimeError: if the store has not been built yet.
    """
    store = _open_store()
    return store.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int = 4) -> str:
    """Retrieve the top-k chunks for ``query`` as a single grounding string.

    Each chunk is prefixed with its source filename and chunks are joined by a
    horizontal-rule separator. Returns an empty string if nothing is found.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    if not docs:
        return ""

    blocks: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[source: {source}]\n{doc.page_content}")
    return _CHUNK_SEPARATOR.join(blocks)
