"""The AgentForge LangGraph graph: router + specialists + persistent memory.

CONTRACT (consumed by cli.py and eval/evaluate.py):
    - build_graph(checkpointer=None) -> CompiledStateGraph
        Build and compile the StateGraph over AgentState with:
          * node "router": LLM classifier that sets state["route"] to one of
            "tool_agent" | "rag_agent" | "chat_agent". It must NOT answer.
          * conditional edges from "router" to the three specialist entrypoints
            based on state["route"].
          * tool_agent path: binds agentforge.tools.TOOLS, uses a ToolNode loop
            so the model can call calculator / unit_converter, and records every
            invoked tool name into state["tool_calls_made"].
          * rag_agent: calls agentforge.rag.retrieve(), stores the text in
            state["retrieved_context"], and answers grounded in it.
          * chat_agent: answers directly (relies on conversation memory).
        If `checkpointer` is None, compile with a default SqliteSaver at
        config.CHECKPOINT_DB so memory persists across turns AND process runs.
    - run_turn(graph, user_input: str, thread_id: str) -> dict
        Invoke the graph for one turn on the given thread and return a dict:
          {"answer": str, "route": str, "tool_calls_made": list[str],
           "retrieved_context": str | None, "state": <final state>}

Implemented by the graph agent in Phase 2.
"""

from __future__ import annotations

# TODO(Phase 2 - graph agent): implement build_graph() and run_turn().
