"""Tool definitions for the tool-calling agent.

CONTRACT (do not change without updating state.py + scenarios + eval):
    - Expose two LangChain tools whose `.name` are exactly:
        "calculator"      -> evaluate an arithmetic expression
        "unit_converter"  -> convert a value between supported units
    - Expose `TOOLS: list` containing both tool objects, in that order.

Implemented by the tools agent in Phase 1.
"""

from __future__ import annotations

# TODO(Phase 1 - tools agent): implement `calculator` and `unit_converter`
# as langchain_core.tools tools, then populate TOOLS.
TOOLS: list = []
