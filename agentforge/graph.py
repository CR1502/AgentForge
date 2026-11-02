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

----------------------------------------------------------------------------
Design notes / decisions
----------------------------------------------------------------------------
* tool_calls_made accounting is PER-TURN. run_turn resets the key to [] at the
  start of every invocation (and route/retrieved_context likewise), so the
  values returned describe exactly the work done on THIS turn. The evaluation
  harness inspects the final turn, so per-turn accounting is the correct and
  simplest choice. Conversation MEMORY (the `messages` list) is NOT reset -- it
  is reduced by `add_messages` and persisted by the checkpointer across turns
  and across process runs.
* SqliteSaver is constructed from an explicit, long-lived sqlite3 connection
  (check_same_thread=False) rather than the `from_conn_string` context manager,
  because the context manager would close the DB when its `with` block exits --
  before the compiled graph ever runs. The connection here lives for the life
  of the process.
* The router uses `with_structured_output`; if that fails for any reason we
  default to chat_agent so a turn never crashes the graph.
"""

from __future__ import annotations

import sqlite3
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from agentforge import config
from agentforge.rag import retrieve
from agentforge.state import AgentState
from agentforge.tools import TOOLS

# ---------------------------------------------------------------------------
# LLM construction (lazy singleton so build_graph() needs no network call)
# ---------------------------------------------------------------------------

_LLM = None


def _get_llm():
    """Return a process-wide ChatOllama client, constructing it on first use.

    Construction makes no network call -- the cold load happens on the first
    `.invoke()` -- so building/compiling the graph stays offline.
    """
    global _LLM
    if _LLM is None:
        from langchain_ollama import ChatOllama

        _LLM = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )
    return _LLM


# ---------------------------------------------------------------------------
# Router structured-output schema
# ---------------------------------------------------------------------------


class RouteDecision(BaseModel):
    """Structured classification of a single user turn into one specialist."""

    route: Literal["tool_agent", "rag_agent", "chat_agent"] = Field(
        description=(
            "Which specialist should handle the user's latest message: "
            "'tool_agent' for arithmetic/math or unit conversions; "
            "'rag_agent' for questions about the Helix Dynamics company "
            "handbook / knowledge base; 'chat_agent' for greetings, general "
            "chit-chat, or anything else."
        )
    )


_ROUTER_SYSTEM = (
    "You are a routing classifier for a multi-agent assistant. Using the "
    "latest user message (with the earlier conversation only as context), "
    "choose exactly one route:\n"
    "- tool_agent: the user wants a calculation -- arithmetic (add, subtract, "
    "multiply, divide, powers, remainders) or a unit conversion (length, mass/"
    "weight, or temperature).\n"
    "- rag_agent: the user asks about facts in the HELIX DYNAMICS company "
    "handbook / internal knowledge base -- company policies, benefits, PTO, "
    "procedures, org details. Choose this ONLY for questions about the company "
    "itself.\n"
    "- chat_agent: greetings, small talk, and -- importantly -- any question "
    "that must be answered from the ongoing conversation or the user's OWN "
    "earlier statements (e.g. 'what is my project codename?', 'what did I say "
    "my name was?'). Personal recall and anything not about Helix Dynamics or a "
    "calculation goes here.\n"
    "Respond with the route only; do not answer the user's question."
)


# System prompts for the specialists.
_TOOL_SYSTEM = (
    "You are a helpful assistant with access to a `calculator` tool (arithmetic) "
    "and a `unit_converter` tool (length/mass/temperature conversions). When the "
    "user needs a computation or conversion, call the appropriate tool rather "
    "than answering from memory. After you receive tool results, reply to the "
    "user in clear natural language that states the final answer."
)

_RAG_SYSTEM = (
    "You are a company-knowledge assistant for Helix Dynamics. Answer the user's "
    "question using ONLY the provided context below. If the context does not "
    "contain the answer, say you don't know -- do not invent facts.\n\n"
    "Context:\n{context}"
)

_CHAT_SYSTEM = (
    "You are AgentForge, a friendly, concise conversational assistant. Answer "
    "directly using the conversation so far. You have no tools and no document "
    "retrieval; rely on the dialogue history (memory)."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_user_text(messages) -> str:
    """Return the content of the most recent human message (or "")."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _recent_context(messages, max_prior: int = 4) -> str:
    """Render the last few messages BEFORE the current turn as plain text.

    Gives the router enough conversational context to tell personal recall
    (chat_agent) from a company-handbook question (rag_agent), without dumping
    the full history.
    """
    # Everything except the final (current) human message.
    prior = list(messages[:-1])[-max_prior:]
    lines = []
    for msg in prior:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        content = content.strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _router_node(state: AgentState) -> dict:
    """Classify the latest user turn into a route. Produces NO answer message."""
    messages = state["messages"]
    user_text = _latest_user_text(messages)
    context = _recent_context(messages)
    classifier_input = user_text
    if context:
        classifier_input = (
            f"Conversation so far:\n{context}\n\n"
            f"Latest user message to classify: {user_text}"
        )
    try:
        classifier = _get_llm().with_structured_output(RouteDecision)
        decision = classifier.invoke(
            [
                SystemMessage(content=_ROUTER_SYSTEM),
                HumanMessage(content=classifier_input),
            ]
        )
        route = decision.route
    except Exception:
        # Be resilient: never let a routing hiccup crash the turn.
        route = "chat_agent"
    return {"route": route}


def _route_edge(state: AgentState) -> str:
    """Conditional-edge selector: map state['route'] to a specialist node."""
    route = state.get("route")
    if route == "tool_agent":
        return "tool_agent"
    if route == "rag_agent":
        return "rag_agent"
    return "chat_agent"


def _tool_agent_node(state: AgentState) -> dict:
    """Let the model either call a tool or produce the final answer.

    Every tool the model requests on this hop is appended (by name) to
    state['tool_calls_made'] so the eval harness can score tool-call accuracy.
    """
    messages = state["messages"]
    # Prepend the tool-usage system prompt (not persisted to memory; it is only
    # used to condition this generation).
    llm_with_tools = _get_llm().bind_tools(TOOLS)
    response = llm_with_tools.invoke([SystemMessage(content=_TOOL_SYSTEM)] + list(messages))

    update: dict = {"messages": [response]}

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        invoked = [tc["name"] for tc in tool_calls]
        update["tool_calls_made"] = list(state.get("tool_calls_made", [])) + invoked

    return update


def _tool_agent_edge(state: AgentState) -> str:
    """After tool_agent: run the tools if any were requested, else finish."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def _rag_agent_node(state: AgentState) -> dict:
    """Retrieve grounding context and answer strictly from it."""
    question = _latest_user_text(state["messages"])
    context = retrieve(question, k=4)

    system = _RAG_SYSTEM.format(context=context if context else "(no context found)")
    response = _get_llm().invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=question),
        ]
    )
    return {"retrieved_context": context, "messages": [response]}


def _chat_agent_node(state: AgentState) -> dict:
    """Answer directly from the conversation (memory only, no tools/retrieval)."""
    response = _get_llm().invoke(
        [SystemMessage(content=_CHAT_SYSTEM)] + list(state["messages"])
    )
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

# Module-level handle to the default checkpointer's connection so it (and the
# SQLite file handle) stays alive for the life of the process.
_DEFAULT_SAVER: Optional[SqliteSaver] = None


def _default_checkpointer() -> SqliteSaver:
    """Build (once) a persistent SqliteSaver at config.CHECKPOINT_DB.

    Uses an explicit sqlite3 connection with check_same_thread=False so the
    saver works regardless of which thread LangGraph invokes it on, and so the
    DB is NOT closed out from under the running graph (unlike the
    `from_conn_string` context-manager pattern).
    """
    global _DEFAULT_SAVER
    if _DEFAULT_SAVER is None:
        conn = sqlite3.connect(str(config.CHECKPOINT_DB), check_same_thread=False)
        _DEFAULT_SAVER = SqliteSaver(conn)
    return _DEFAULT_SAVER


def build_graph(checkpointer=None):
    """Build and compile the AgentForge graph over AgentState.

    Args:
        checkpointer: A LangGraph checkpointer. If None, a persistent
            SqliteSaver at config.CHECKPOINT_DB is created so memory survives
            across turns and across process runs. Pass an in-memory saver
            (e.g. from the eval harness) to isolate runs.

    Returns:
        The compiled StateGraph. Compiling makes no LLM call.
    """
    builder = StateGraph(AgentState)

    builder.add_node("router", _router_node)
    builder.add_node("tool_agent", _tool_agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("rag_agent", _rag_agent_node)
    builder.add_node("chat_agent", _chat_agent_node)

    builder.add_edge(START, "router")

    # Router fans out to one specialist based on state["route"].
    builder.add_conditional_edges(
        "router",
        _route_edge,
        {
            "tool_agent": "tool_agent",
            "rag_agent": "rag_agent",
            "chat_agent": "chat_agent",
        },
    )

    # tool_agent <-> tools loop: generate -> (maybe) run tools -> generate ...
    builder.add_conditional_edges(
        "tool_agent",
        _tool_agent_edge,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "tool_agent")

    # RAG and chat are single-shot answers.
    builder.add_edge("rag_agent", END)
    builder.add_edge("chat_agent", END)

    if checkpointer is None:
        checkpointer = _default_checkpointer()

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Single-turn driver
# ---------------------------------------------------------------------------


def run_turn(graph, user_input: str, thread_id: str) -> dict:
    """Run one conversational turn on `thread_id` and summarise the result.

    `route`, `retrieved_context`, and `tool_calls_made` are reset at the start
    of the turn so the returned values describe THIS turn only; `messages`
    accumulates across turns via the checkpointer (that is the memory).
    """
    config_ = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "route": None,
        "retrieved_context": None,
        "tool_calls_made": [],
    }

    final_state = graph.invoke(inputs, config=config_)

    # The final assistant message is the last message in the state.
    answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = msg.content
            answer = content if isinstance(content, str) else str(content)
            break

    return {
        "answer": answer,
        "route": final_state.get("route"),
        "tool_calls_made": final_state.get("tool_calls_made", []),
        "retrieved_context": final_state.get("retrieved_context"),
        "state": final_state,
    }
