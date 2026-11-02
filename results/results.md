# AgentForge Evaluation Results

Run on **2026-11-02T16:52:22.629368+00:00** against **22 held-out scenarios** using local **gemma4:12b** (via Ollama, temperature 0.0). Wall-clock time: **657.3 s**.

## Headline metrics

| Metric | Score | Percentage |
| --- | --- | --- |
| Routing accuracy | 22/22 | 100.0% |
| Tool-call accuracy | 22/22 | 100.0% |
| Task completion rate | 21/22 | 95.5% |

## Run metadata

- **Chat model:** gemma4:12b
- **Embedding model:** nomic-embed-text
- **Ollama base URL:** http://localhost:11434
- **Temperature:** 0.0
- **Scenarios:** 22
- **Timestamp (UTC):** 2026-11-02T16:52:22.629368+00:00
- **Wall-clock seconds:** 657.3

## Per-category breakdown

### Routing & completion by expected route

| Expected route | N | Routing accuracy | Task completion |
| --- | --- | --- | --- |
| chat_agent | 5 | 5/5 (100.0%) | 5/5 (100.0%) |
| rag_agent | 8 | 8/8 (100.0%) | 7/8 (87.5%) |
| tool_agent | 9 | 9/9 (100.0%) | 9/9 (100.0%) |

### Tool-call accuracy

| Subset | N | Correct | Accuracy |
| --- | --- | --- | --- |
| Tool expected | 9 | 9 | 100.0% |
| No tool expected | 13 | 13 | 100.0% |

## Per-scenario results

| ID | Expected route | Actual route | Route | Expected tool | Actual tools | Tool | Completion |
| --- | --- | --- | :-: | --- | --- | :-: | :-: |
| calc-divide-add | tool_agent | tool_agent | ✓ | calculator | calculator | ✓ | ✓ |
| calc-parenthesized-multiply | tool_agent | tool_agent | ✓ | calculator | calculator | ✓ | ✓ |
| calc-power-subtract | tool_agent | tool_agent | ✓ | calculator | calculator | ✓ | ✓ |
| calc-subtract-product | tool_agent | tool_agent | ✓ | calculator | calculator | ✓ | ✓ |
| calc-percentage-decimal | tool_agent | tool_agent | ✓ | calculator | calculator | ✓ | ✓ |
| unit-km-to-miles | tool_agent | tool_agent | ✓ | unit_converter | unit_converter | ✓ | ✓ |
| unit-celsius-to-fahrenheit | tool_agent | tool_agent | ✓ | unit_converter | unit_converter | ✓ | ✓ |
| unit-kg-to-pounds | tool_agent | tool_agent | ✓ | unit_converter | unit_converter | ✓ | ✓ |
| unit-miles-to-km | tool_agent | tool_agent | ✓ | unit_converter | unit_converter | ✓ | ✓ |
| rag-pto-days | rag_agent | rag_agent | ✓ | — | — | ✓ | ✓ |
| rag-ceo-name | rag_agent | rag_agent | ✓ | — | — | ✓ | ✓ |
| rag-founding-city | rag_agent | rag_agent | ✓ | — | — | ✓ | ✓ |
| rag-password-rotation | rag_agent | rag_agent | ✓ | — | — | ✓ | ✓ |
| rag-platinum-sev1-response | rag_agent | rag_agent | ✓ | — | — | ✓ | ✓ |
| rag-startup-plan-price | rag_agent | rag_agent | ✓ | — | — | ✓ | ✓ |
| rag-oncall-stipend | rag_agent | rag_agent | ✓ | — | — | ✓ | ✗ |
| chat-greeting | chat_agent | chat_agent | ✓ | — | — | ✓ | ✓ |
| chat-octopus-fact | chat_agent | chat_agent | ✓ | — | — | ✓ | ✓ |
| chat-opinion-language | chat_agent | chat_agent | ✓ | — | — | ✓ | ✓ |
| memory-chat-codename | chat_agent | chat_agent | ✓ | — | — | ✓ | ✓ |
| memory-chat-city | chat_agent | chat_agent | ✓ | — | — | ✓ | ✓ |
| memory-context-then-rag | rag_agent | rag_agent | ✓ | — | — | ✓ | ✓ |

## Failures

- **rag-oncall-stipend** — completion (missing ['750'])
  - Answer: "I don't know."

## How these numbers were produced

Every number above comes from a single real run of `scripts/run_eval.py` against all 22 held-out scenarios. The graph was driven locally by **gemma4:12b** served through Ollama at temperature 0.0 (deterministic), with **nomic-embed-text** embeddings for retrieval. Each scenario ran on its own in-memory conversation thread (`eval-<id>`), so multi-turn scenarios exercise memory while scenarios stay isolated from one another. No number was hand-edited or synthesized.
