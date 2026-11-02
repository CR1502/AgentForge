"""Central configuration for AgentForge.

All paths are resolved relative to the repository root so the project runs the
same way regardless of the current working directory. Everything runs locally
against Ollama -- there are no API keys.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root = parent of the `agentforge` package directory.
ROOT = Path(__file__).resolve().parent.parent

# ---- Models (served by local Ollama) ---------------------------------------
# Chat/tool-calling model. gemma4:12b advertises the `tools` capability.
LLM_MODEL = os.environ.get("AGENTFORGE_LLM_MODEL", "gemma4:12b")
# Dedicated embedding model for the RAG vector store.
EMBED_MODEL = os.environ.get("AGENTFORGE_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Deterministic generation for reproducible evaluation runs.
LLM_TEMPERATURE = float(os.environ.get("AGENTFORGE_TEMPERATURE", "0.0"))

# ---- Paths ------------------------------------------------------------------
CORPUS_DIR = ROOT / "data" / "corpus"
CHROMA_DIR = ROOT / ".chroma"              # persistent vector store (git-ignored)
COLLECTION_NAME = "agentforge_corpus"
CHECKPOINT_DB = ROOT / ".agentforge_memory.sqlite"  # memory (git-ignored)

SCENARIOS_PATH = ROOT / "eval" / "scenarios.json"
RESULTS_JSON = ROOT / "results" / "results.json"
RESULTS_MD = ROOT / "results" / "results.md"
