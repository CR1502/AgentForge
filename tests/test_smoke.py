"""Lightweight structural smoke tests (no LLM calls required).

These verify the package wiring/contracts so breakage is caught before the
(slow) LLM evaluation run. Run with: python -m pytest tests/ -q
"""

from __future__ import annotations


def test_state_taxonomy():
    from agentforge.state import AgentState  # noqa: F401
    from agentforge import state

    assert set(state.Route.__args__) == {"tool_agent", "rag_agent", "chat_agent"}


def test_tools_contract():
    from agentforge.tools import TOOLS

    names = {t.name for t in TOOLS}
    assert names == {"calculator", "unit_converter"}


def test_graph_builds():
    # Compiling the graph must not require a live LLM.
    from agentforge.graph import build_graph

    graph = build_graph()
    assert graph is not None
