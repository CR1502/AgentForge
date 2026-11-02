"""Interactive multi-turn chat over the AgentForge graph.

Usage:  python -m agentforge.cli            # new conversation
        python -m agentforge.cli --thread my-thread   # resume a thread

Demonstrates persistent memory: re-running with the same --thread resumes the
prior conversation (loaded from the SqliteSaver checkpointer).

CONTRACT: a `main()` entrypoint that loops reading stdin, calls
graph.run_turn(), and prints the answer plus the route/tool trace.

Implemented by the graph agent in Phase 2.
"""

from __future__ import annotations

# TODO(Phase 2 - graph agent): implement main().
