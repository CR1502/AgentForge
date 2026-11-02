"""Evaluation harness: run the graph against held-out scenarios and score it.

CONTRACT (consumed by scripts/run_eval.py):
    - load_scenarios(path=config.SCENARIOS_PATH) -> list[dict]
    - evaluate(graph, scenarios) -> dict
        Run every scenario on its own thread_id (multi-turn scenarios replay
        their turns in order on one thread, so memory is exercised). For each
        scenario record: actual route, actual tool calls, final answer, and the
        per-metric pass/fail. Compute and return aggregate metrics:
          * routing_accuracy      : fraction with actual route == expected_route
          * tool_call_accuracy    : fraction where the set of tools actually
                                    called matches the expected tool (incl. the
                                    "no tool expected -> no tool called" case)
          * task_completion_rate  : fraction whose final answer satisfies the
                                    scenario's completion check
        plus per-scenario detail records.
    - write_results(results, json_path=config.RESULTS_JSON,
                    md_path=config.RESULTS_MD) -> None
        Persist results.json (machine-readable) and results.md (human-readable
        table + headline numbers). EVERY number must come from the real run.

SCENARIO SCHEMA (see eval/scenarios.json):
    {
      "id": "unique-id",
      "description": "what this tests",
      "turns": ["first user message", "second user message", ...],
      "expected_route": "tool_agent" | "rag_agent" | "chat_agent",
      "expected_tool": "calculator" | "unit_converter" | null,
      "expected_answer_contains": ["substring", ...]   # case-insensitive, all
    }
    (expected_route / expected_tool refer to the FINAL turn.)

Implemented by the eval agent in Phase 3.
"""

from __future__ import annotations

# TODO(Phase 3 - eval agent): implement load_scenarios / evaluate / write_results.
