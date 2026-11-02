"""Entrypoint: run the full evaluation and write results.{json,md}.

Usage:  python scripts/run_eval.py

Builds the graph (with a throwaway in-memory checkpointer per scenario thread),
runs eval.evaluate over all scenarios, prints the headline metrics, and writes
results/results.json + results/results.md.

Implemented by the eval agent in Phase 3.
"""

from __future__ import annotations

# TODO(Phase 3 - eval agent): wire build_graph -> evaluate -> write_results.
