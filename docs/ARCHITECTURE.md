# AgentForge — Architecture & Contracts

This is the internal contract shared by all modules. Keep it in sync with
`agentforge/state.py`.

## Graph topology

```
                 ┌────────────┐
   user turn ──▶ │   router   │  (LLM classifier; sets state["route"])
                 └─────┬──────┘
          conditional edge on state["route"]
        ┌──────────────┼───────────────┐
        ▼              ▼                ▼
 ┌────────────┐ ┌────────────┐  ┌────────────┐
 │ tool_agent │ │  rag_agent │  │ chat_agent │
 │ calc /     │ │ Chroma     │  │ direct +   │
 │ unit_conv  │ │ retrieval  │  │ memory     │
 └─────┬──────┘ └─────┬──────┘  └─────┬──────┘
       │ (ToolNode     │               │
       │  loop)        │               │
       └──────────────▶END◀────────────┘
```

- **State**: `agentforge.state.AgentState` (messages + route + retrieved_context
  + tool_calls_made).
- **Memory**: LangGraph `SqliteSaver` checkpointer keyed on `thread_id`.
  Persists across turns within a conversation and across process restarts.
- **LLM**: local Ollama `gemma4:12b` (native tool-calling).
- **Embeddings**: local Ollama `nomic-embed-text`.
- **Vector store**: persistent Chroma collection at `.chroma/`.

## Routing taxonomy
`tool_agent` | `rag_agent` | `chat_agent` (see `state.Route`).

## Tool taxonomy
`calculator` | `unit_converter` (see `state.ToolName`). Tool `.name` values must
match exactly.

## Scenario schema (eval/scenarios.json)
```json
{
  "id": "string",
  "description": "string",
  "turns": ["message 1", "message 2"],
  "expected_route": "tool_agent | rag_agent | chat_agent",
  "expected_tool": "calculator | unit_converter | null",
  "expected_answer_contains": ["substring", "..."]
}
```
`expected_route` / `expected_tool` describe the FINAL turn. `expected_answer_contains`
is matched case-insensitively against the final answer; ALL substrings must appear.

## Metrics
- **routing_accuracy** — actual route == expected_route.
- **tool_call_accuracy** — tools actually called match the expected tool,
  including the "none expected → none called" case.
- **task_completion_rate** — final answer satisfies `expected_answer_contains`.
