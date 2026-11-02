"""Entrypoint: run the full evaluation and write results.{json,md}.

Usage:  python scripts/run_eval.py

Builds the graph (with a throwaway in-memory checkpointer per scenario thread),
runs eval.evaluate over all scenarios, prints the headline metrics, and writes
results/results.json + results/results.md.

Implemented by the eval agent in Phase 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import agentforge...` / `import eval...` work when run as
# `python scripts/run_eval.py` from the repo root, without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agentforge import config  # noqa: E402
from agentforge.graph import build_graph  # noqa: E402
from eval.evaluate import evaluate, load_scenarios, write_results  # noqa: E402


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> int:
    # In-memory checkpointer so eval runs are repeatable and never touch the
    # real persistent memory DB. The graph is built ONCE; each scenario uses a
    # unique thread_id for isolation.
    graph = build_graph(checkpointer=InMemorySaver())

    scenarios = load_scenarios()
    print(
        f"Running evaluation: {len(scenarios)} scenarios against "
        f"{config.LLM_MODEL} via Ollama ({config.OLLAMA_BASE_URL}).",
        flush=True,
    )
    print("The first call is a cold model load; the full run may take minutes.", flush=True)

    results = evaluate(graph, scenarios)

    m = results["metrics"]
    meta = results["metadata"]
    n = m["num_scenarios"]

    print("\n=== Headline metrics ===", flush=True)
    print(
        f"Routing accuracy     : {m['routing_passed']}/{n}  "
        f"({_fmt_pct(m['routing_accuracy'])})"
    )
    print(
        f"Tool-call accuracy   : {m['tool_call_passed']}/{n}  "
        f"({_fmt_pct(m['tool_call_accuracy'])})"
    )
    print(
        f"Task completion rate : {m['completion_passed']}/{n}  "
        f"({_fmt_pct(m['task_completion_rate'])})"
    )
    print(f"Total wall-clock time: {meta['wall_clock_seconds']} s", flush=True)

    write_results(results)
    print(f"\nWrote {config.RESULTS_JSON}", flush=True)
    print(f"Wrote {config.RESULTS_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
