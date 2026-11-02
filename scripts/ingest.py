"""Build (or rebuild) the Chroma vector store from the document corpus.

Usage:  python scripts/ingest.py            # build if missing
        python scripts/ingest.py --reset    # wipe and rebuild

CONTRACT: calls agentforge.rag.build_vectorstore() and prints a short summary
(number of source files, number of chunks, persist location).

Implemented by the RAG agent in Phase 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import agentforge...` work when run as `python scripts/ingest.py`
# from the repo root, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentforge import config  # noqa: E402
from agentforge.rag import build_vectorstore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or rebuild the AgentForge Chroma vector store."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete any existing collection/dir and rebuild from scratch.",
    )
    args = parser.parse_args()

    source_files = sorted(config.CORPUS_DIR.glob("*.md"))

    action = "Rebuilding (reset)" if args.reset else "Building"
    print(f"{action} vector store from {config.CORPUS_DIR} ...")

    store = build_vectorstore(reset=args.reset)

    try:
        num_chunks = store._collection.count()
    except Exception:
        num_chunks = "unknown"

    print("Ingest complete.")
    print(f"  Source files : {len(source_files)}")
    print(f"  Chunks        : {num_chunks}")
    print(f"  Collection    : {config.COLLECTION_NAME}")
    print(f"  Persisted to  : {config.CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
