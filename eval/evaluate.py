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

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agentforge import config
from agentforge.graph import run_turn

# Answers can be long; keep enough in the JSON to verify completion by eye.
_ANSWER_TRUNCATE = 300


def load_scenarios(path=config.SCENARIOS_PATH) -> list[dict]:
    """Load the held-out evaluation scenarios from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        scenarios = json.load(fh)
    if not isinstance(scenarios, list):
        raise ValueError(f"Expected a list of scenarios in {path}")
    return scenarios


def _truncate(text: str, limit: int = _ANSWER_TRUNCATE) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _completion_pass(answer: str, expected_substrings: list[str]) -> bool:
    """All expected substrings present (case-insensitive); empty -> non-empty."""
    if not expected_substrings:
        return bool(answer and answer.strip())
    lowered = (answer or "").lower()
    return all(sub.lower() in lowered for sub in expected_substrings)


def evaluate(graph, scenarios: list[dict]) -> dict:
    """Replay every scenario against the graph and score it. All real numbers."""
    started_at = datetime.now(timezone.utc)
    wall_start = time.monotonic()

    details: list[dict] = []

    for scenario in scenarios:
        scenario_id = scenario["id"]
        thread_id = f"eval-{scenario_id}"
        turns = scenario["turns"]

        expected_route = scenario.get("expected_route")
        expected_tool = scenario.get("expected_tool")
        expected_contains = scenario.get("expected_answer_contains", []) or []
        expected_tool_set = {expected_tool} if expected_tool else set()

        # Replay all turns on the same thread; capture the FINAL turn's result.
        per_turn_trace: list[dict] = []
        final = None
        for turn_text in turns:
            final = run_turn(graph, turn_text, thread_id)
            per_turn_trace.append(
                {
                    "input": turn_text,
                    "route": final["route"],
                    "tool_calls_made": list(final["tool_calls_made"]),
                }
            )

        actual_route = final["route"]
        actual_tools = list(final["tool_calls_made"])
        actual_tool_set = set(actual_tools)
        actual_answer = final["answer"] or ""

        routing_pass = actual_route == expected_route
        tool_call_pass = actual_tool_set == expected_tool_set
        completion_pass = _completion_pass(actual_answer, expected_contains)

        details.append(
            {
                "id": scenario_id,
                "description": scenario.get("description", ""),
                "turns": turns,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "routing_pass": routing_pass,
                "expected_tool": expected_tool,
                "actual_tools": actual_tools,
                "tool_call_pass": tool_call_pass,
                "expected_answer_contains": expected_contains,
                "actual_answer": _truncate(actual_answer),
                "completion_pass": completion_pass,
                "turn_trace": per_turn_trace,
            }
        )

    wall_seconds = time.monotonic() - wall_start
    n = len(details)

    def _mean(key: str) -> float:
        if n == 0:
            return 0.0
        return sum(1 for d in details if d[key]) / n

    metrics = {
        "routing_accuracy": _mean("routing_pass"),
        "tool_call_accuracy": _mean("tool_call_pass"),
        "task_completion_rate": _mean("completion_pass"),
        "routing_passed": sum(1 for d in details if d["routing_pass"]),
        "tool_call_passed": sum(1 for d in details if d["tool_call_pass"]),
        "completion_passed": sum(1 for d in details if d["completion_pass"]),
        "num_scenarios": n,
    }

    breakdowns = _compute_breakdowns(details)

    metadata = {
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "temperature": config.LLM_TEMPERATURE,
        "num_scenarios": n,
        "timestamp": started_at.isoformat(),
        "wall_clock_seconds": round(wall_seconds, 1),
    }

    return {
        "metrics": metrics,
        "breakdowns": breakdowns,
        "metadata": metadata,
        "scenarios": details,
    }


def _compute_breakdowns(details: list[dict]) -> dict:
    """Per-category honest breakdowns."""
    # Group routing/completion accuracy by expected route.
    by_route: dict[str, dict] = {}
    for d in details:
        route = d["expected_route"]
        bucket = by_route.setdefault(
            route,
            {"count": 0, "routing_passed": 0, "completion_passed": 0},
        )
        bucket["count"] += 1
        bucket["routing_passed"] += int(d["routing_pass"])
        bucket["completion_passed"] += int(d["completion_pass"])
    for bucket in by_route.values():
        c = bucket["count"]
        bucket["routing_accuracy"] = bucket["routing_passed"] / c if c else 0.0
        bucket["completion_rate"] = bucket["completion_passed"] / c if c else 0.0

    # Tool-call accuracy split: tool-expected vs no-tool-expected.
    tool_expected = [d for d in details if d["expected_tool"]]
    no_tool_expected = [d for d in details if not d["expected_tool"]]

    def _tool_bucket(items: list[dict]) -> dict:
        c = len(items)
        passed = sum(1 for d in items if d["tool_call_pass"])
        return {
            "count": c,
            "passed": passed,
            "accuracy": passed / c if c else 0.0,
        }

    return {
        "by_expected_route": by_route,
        "tool_call": {
            "tool_expected": _tool_bucket(tool_expected),
            "no_tool_expected": _tool_bucket(no_tool_expected),
        },
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _check(flag: bool) -> str:
    return "✓" if flag else "✗"


def write_results(results: dict, json_path=config.RESULTS_JSON, md_path=config.RESULTS_MD) -> None:
    """Persist results.json (full dict) and results.md (human-readable report)."""
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    md = _render_markdown(results)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)


def _render_markdown(results: dict) -> str:
    m = results["metrics"]
    meta = results["metadata"]
    bd = results["breakdowns"]
    scenarios = results["scenarios"]
    n = m["num_scenarios"]

    lines: list[str] = []
    lines.append("# AgentForge Evaluation Results")
    lines.append("")
    lines.append(
        f"Run on **{meta['timestamp']}** against **{n} held-out scenarios** "
        f"using local **{meta['llm_model']}** (via Ollama, temperature "
        f"{meta['temperature']}). Wall-clock time: "
        f"**{meta['wall_clock_seconds']} s**."
    )
    lines.append("")

    # Headline metrics.
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Score | Percentage |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Routing accuracy | {m['routing_passed']}/{n} | "
        f"{_fmt_pct(m['routing_accuracy'])} |"
    )
    lines.append(
        f"| Tool-call accuracy | {m['tool_call_passed']}/{n} | "
        f"{_fmt_pct(m['tool_call_accuracy'])} |"
    )
    lines.append(
        f"| Task completion rate | {m['completion_passed']}/{n} | "
        f"{_fmt_pct(m['task_completion_rate'])} |"
    )
    lines.append("")

    # Metadata.
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- **Chat model:** {meta['llm_model']}")
    lines.append(f"- **Embedding model:** {meta['embed_model']}")
    lines.append(f"- **Ollama base URL:** {meta['ollama_base_url']}")
    lines.append(f"- **Temperature:** {meta['temperature']}")
    lines.append(f"- **Scenarios:** {meta['num_scenarios']}")
    lines.append(f"- **Timestamp (UTC):** {meta['timestamp']}")
    lines.append(f"- **Wall-clock seconds:** {meta['wall_clock_seconds']}")
    lines.append("")

    # Breakdowns.
    lines.append("## Per-category breakdown")
    lines.append("")
    lines.append("### Routing & completion by expected route")
    lines.append("")
    lines.append("| Expected route | N | Routing accuracy | Task completion |")
    lines.append("| --- | --- | --- | --- |")
    for route in sorted(bd["by_expected_route"]):
        b = bd["by_expected_route"][route]
        lines.append(
            f"| {route} | {b['count']} | "
            f"{b['routing_passed']}/{b['count']} ({_fmt_pct(b['routing_accuracy'])}) | "
            f"{b['completion_passed']}/{b['count']} ({_fmt_pct(b['completion_rate'])}) |"
        )
    lines.append("")
    lines.append("### Tool-call accuracy")
    lines.append("")
    lines.append("| Subset | N | Correct | Accuracy |")
    lines.append("| --- | --- | --- | --- |")
    te = bd["tool_call"]["tool_expected"]
    nte = bd["tool_call"]["no_tool_expected"]
    lines.append(
        f"| Tool expected | {te['count']} | {te['passed']} | {_fmt_pct(te['accuracy'])} |"
    )
    lines.append(
        f"| No tool expected | {nte['count']} | {nte['passed']} | "
        f"{_fmt_pct(nte['accuracy'])} |"
    )
    lines.append("")

    # Per-scenario table.
    lines.append("## Per-scenario results")
    lines.append("")
    lines.append(
        "| ID | Expected route | Actual route | Route | Expected tool | "
        "Actual tools | Tool | Completion |"
    )
    lines.append("| --- | --- | --- | :-: | --- | --- | :-: | :-: |")
    for d in scenarios:
        expected_tool = d["expected_tool"] or "—"
        actual_tools = ", ".join(d["actual_tools"]) if d["actual_tools"] else "—"
        lines.append(
            f"| {d['id']} | {d['expected_route']} | {d['actual_route']} | "
            f"{_check(d['routing_pass'])} | {expected_tool} | {actual_tools} | "
            f"{_check(d['tool_call_pass'])} | {_check(d['completion_pass'])} |"
        )
    lines.append("")

    # Failures detail (honest picture).
    failures = [
        d
        for d in scenarios
        if not (d["routing_pass"] and d["tool_call_pass"] and d["completion_pass"])
    ]
    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("None — every scenario passed all three metrics.")
    else:
        for d in failures:
            failed_metrics = []
            if not d["routing_pass"]:
                failed_metrics.append(
                    f"routing (expected {d['expected_route']}, got {d['actual_route']})"
                )
            if not d["tool_call_pass"]:
                failed_metrics.append(
                    f"tool-call (expected {d['expected_tool'] or 'none'}, "
                    f"got {d['actual_tools'] or 'none'})"
                )
            if not d["completion_pass"]:
                failed_metrics.append(
                    f"completion (missing {d['expected_answer_contains']})"
                )
            lines.append(f"- **{d['id']}** — {'; '.join(failed_metrics)}")
            lines.append(f"  - Answer: {d['actual_answer']!r}")
    lines.append("")

    # Provenance note.
    lines.append("## How these numbers were produced")
    lines.append("")
    lines.append(
        f"Every number above comes from a single real run of "
        f"`scripts/run_eval.py` against all {n} held-out scenarios. The graph "
        f"was driven locally by **{meta['llm_model']}** served through Ollama at "
        f"temperature {meta['temperature']} (deterministic), with "
        f"**{meta['embed_model']}** embeddings for retrieval. Each scenario ran "
        f"on its own in-memory conversation thread (`eval-<id>`), so multi-turn "
        f"scenarios exercise memory while scenarios stay isolated from one "
        f"another. No number was hand-edited or synthesized."
    )
    lines.append("")

    return "\n".join(lines)
