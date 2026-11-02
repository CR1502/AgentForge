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

import argparse
import sys
import uuid

from agentforge.graph import build_graph, run_turn

# ANSI dim styling for the trace line (degrades gracefully if unsupported).
_DIM = "\033[2m"
_RESET = "\033[0m"


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentforge.cli",
        description="Interactive multi-turn chat over the AgentForge graph.",
    )
    parser.add_argument(
        "--thread",
        default=f"cli-{uuid.uuid4()}",
        help="Conversation thread id. Reuse it to resume a prior conversation.",
    )
    return parser.parse_args(argv)


def _print_trace(result: dict) -> None:
    """Print the dim one-line route/tool trace for a turn."""
    route = result.get("route")
    tools = result.get("tool_calls_made") or []
    tools_str = ", ".join(tools) if tools else "none"
    sys.stdout.write(f"{_DIM}  [route: {route} | tools: {tools_str}]{_RESET}\n")


def main(argv=None) -> int:
    args = _parse_args(argv)
    thread_id = args.thread

    # Persistent-memory graph (default SqliteSaver at config.CHECKPOINT_DB).
    graph = build_graph()

    print("AgentForge interactive chat.")
    print(f"Thread id: {thread_id}")
    print(f"Resume this conversation later with:  --thread {thread_id}")
    print("Type your message and press Enter. Use 'exit', 'quit', or Ctrl-D to leave.\n")

    while True:
        try:
            line = input("you> ")
        except EOFError:
            # Ctrl-D / end of piped input.
            print()
            break
        except KeyboardInterrupt:
            # Ctrl-C: exit cleanly.
            print("\nGoodbye.")
            break

        user_input = line.strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            result = run_turn(graph, user_input, thread_id)
        except Exception as exc:  # keep the loop alive on a single-turn failure
            print(f"[error] {exc}\n")
            continue

        print(f"bot> {result['answer']}")
        _print_trace(result)
        print()

    print(f"(resume with: --thread {thread_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
