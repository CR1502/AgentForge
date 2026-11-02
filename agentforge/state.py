"""Shared state contract for the AgentForge LangGraph graph.

This module is the single source of truth for the graph's state schema and the
routing/tool taxonomy. Every node, the evaluation harness, and the scenario
dataset must agree on the literals defined here.

ROUTING TAXONOMY (set by the `router` node, consumed by conditional edges):
    - "tool_agent"  : query needs a computational tool (math / unit conversion)
    - "rag_agent"   : query asks about facts in the document corpus
    - "chat_agent"  : general conversation / anything else (uses memory only)

TOOL TAXONOMY (names must match the `.name` of the tools in agentforge.tools):
    - "calculator"      : arithmetic expression evaluation
    - "unit_converter"  : convert a value between supported units
    - None              : no tool should be called
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Route = Literal["tool_agent", "rag_agent", "chat_agent"]
ToolName = Literal["calculator", "unit_converter"]


class AgentState(TypedDict):
    """State passed between nodes.

    Keys
    ----
    messages:
        The running conversation. Reduced with `add_messages` so each node can
        return only the new messages it produced. Memory persistence across
        turns is provided by the checkpointer keyed on thread_id.
    route:
        The branch chosen by the router. One of the `Route` literals.
    retrieved_context:
        The text retrieved by the rag_agent (if it ran). Used for grounding and
        for eval inspection. None if RAG did not run.
    tool_calls_made:
        Flat list of tool names actually invoked during this turn, in order.
        Populated by the tool-calling path. Used by the evaluation harness to
        score tool-call accuracy. Reduced by replacement (last writer wins per
        node return); nodes should append to the incoming list.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    route: Optional[Route]
    retrieved_context: Optional[str]
    tool_calls_made: List[str]
