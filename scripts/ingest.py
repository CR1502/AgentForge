"""Build (or rebuild) the Chroma vector store from the document corpus.

Usage:  python scripts/ingest.py            # build if missing
        python scripts/ingest.py --reset    # wipe and rebuild

CONTRACT: calls agentforge.rag.build_vectorstore() and prints a short summary
(number of source files, number of chunks, persist location).

Implemented by the RAG agent in Phase 1.
"""

from __future__ import annotations

# TODO(Phase 1 - RAG agent): implement the ingest entrypoint.
