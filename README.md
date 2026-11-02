# AgentForge

A **stateful multi-agent workflow built with [LangGraph](https://langchain-ai.github.io/langgraph/)** — conditional routing, persistent cross-turn memory, real tool-calling, and a RAG retrieval step grounded in a document corpus. It ships with a **real, measured evaluation**: the graph is run against a held-out set of scripted scenarios and scored on routing accuracy, tool-call accuracy, and task-completion rate.

Everything runs **locally against [Ollama](https://ollama.com/)** — no API keys, no cloud calls — so the evaluation is fully reproducible on your own machine.

> **Honesty note.** This is a scoped portfolio project. The evaluation is a **small scripted set (22 scenarios), not a production-scale benchmark**, run against a single local model (`gemma4:12b`). Every number in this README and in [`results/`](results/) comes from an actual run of the code — see [Evaluation](#evaluation). See [Limitations](#limitations) before citing any figure.

---

## What the graph does

A single user turn flows through a **router** that classifies it and then hands off — via **conditional edges** — to one of three specialist agents:

| Route | Agent | What it does |
|---|---|---|
| `tool_agent` | Tool-calling agent | Calls real LangChain tools: `calculator` (AST-safe arithmetic) or `unit_converter` (length / mass / temperature). |
| `rag_agent` | Retrieval agent | Retrieves from a **persistent Chroma vector store** over a company-handbook corpus and answers grounded strictly in the retrieved context. |
| `chat_agent` | Conversational agent | Answers directly, relying on **conversation memory** (no tools, no retrieval). |

**Memory** is provided by a LangGraph `SqliteSaver` checkpointer keyed on `thread_id`, so context is retained **across turns and across process restarts** (verified — see below).

## Architecture

```
                         ┌─────────────┐
      user turn ───────▶ │   router    │   LLM classifier (structured output)
                         │             │   sets state["route"]; writes no answer
                         └──────┬──────┘
                 conditional edge on state["route"]
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   ┌────────────┐        ┌────────────┐        ┌────────────┐
   │ tool_agent │        │  rag_agent │        │ chat_agent │
   │            │        │            │        │            │
   │ bind_tools │        │ retrieve() │        │ direct +   │
   │ calculator │        │  → Chroma  │        │ memory     │
   │ unit_conv. │        │  grounded  │        │            │
   └──────┬─────┘        └──────┬─────┘        └──────┬─────┘
          │ tool_calls?         │                     │
          ▼                     │                     │
   ┌────────────┐               │                     │
   │  ToolNode  │ executes      │                     │
   │ (tools)    │ then loops    │                     │
   └──────┬─────┘ back to agent │                     │
          └─────────────────────┴──────────┬──────────┘
                                            ▼
                                           END

   state: messages (add_messages reducer) · route · retrieved_context · tool_calls_made
   memory: SqliteSaver checkpointer, scoped per thread_id (persists across runs)
```

### Stack
- **Orchestration:** LangGraph `StateGraph` + `ToolNode`, `SqliteSaver` checkpointer
- **LLM:** local Ollama `gemma4:12b` (native tool-calling), temperature 0 for reproducibility
- **Embeddings:** local Ollama `nomic-embed-text`
- **Vector store:** persistent **Chroma** collection over 7 markdown documents
- **Tools:** LangChain `@tool` functions (`calculator`, `unit_converter`)

### Layout
```
agentforge/
  state.py      # AgentState schema + routing/tool taxonomy (the shared contract)
  config.py     # models, paths, all config
  tools.py      # calculator + unit_converter (LangChain tools)
  rag.py        # Chroma vector store: build / retrieve
  graph.py      # the LangGraph graph: router, specialists, memory
  cli.py        # interactive multi-turn chat
data/corpus/    # the document corpus ("Helix Dynamics" handbook, fictional)
eval/
  scenarios.json  # 22 held-out scripted scenarios (expected route/tool/answer)
  evaluate.py     # evaluation engine + metrics
scripts/
  ingest.py     # build the vector store
  run_eval.py   # run the full evaluation -> results/
results/        # results.json + results.md (generated from a real run)
tests/          # structural smoke tests (no LLM required)
docs/           # architecture & contracts
```

> **About the corpus.** It's a **fictional** company handbook ("Helix Dynamics"). That's deliberate: because the facts can't be known from the model's pretraining, a correct RAG answer is evidence that **retrieval actually worked** rather than the model recalling the web.

---

## Setup

### Prerequisites
- [Ollama](https://ollama.com/) installed and running
- [`uv`](https://docs.astral.sh/uv/) (used for the virtualenv)
- Python 3.12 (provisioned by `uv`)

### Install
```bash
# 1. Clone
git clone https://github.com/CR1502/AgentForge.git
cd AgentForge

# 2. Pull the local models (chat + embeddings)
ollama pull gemma4:12b
ollama pull nomic-embed-text

# 3. Create the environment and install dependencies
uv venv --python 3.12
uv pip install -r requirements.txt

# 4. Build the vector store from the corpus
.venv/bin/python scripts/ingest.py --reset
```

A `Makefile` wraps these: `make setup`, `make ingest`, `make chat`, `make eval`, `make test`.

> Dependencies are pinned in [`requirements.txt`](requirements.txt) to the exact versions that produced the committed results (LangGraph/LangChain **1.x**). A full transitive lock is in [`requirements.lock.txt`](requirements.lock.txt).

## Usage

**Interactive chat** (demonstrates routing + persistent memory):
```bash
.venv/bin/python -m agentforge.cli
# ... it prints a thread id; resume that conversation later with:
.venv/bin/python -m agentforge.cli --thread <that-id>
```
Each reply prints a dim trace line showing the chosen route and any tools called.

**Run the evaluation** (regenerates `results/`):
```bash
.venv/bin/python scripts/run_eval.py
```

**Structural smoke tests** (fast, no LLM):
```bash
.venv/bin/python -m pytest tests/ -q
```

---

## Evaluation

### Methodology
- **Held-out set:** 22 scripted scenarios in [`eval/scenarios.json`](eval/scenarios.json), each declaring the *expected* route, the *expected* tool call (or none), and substrings the final answer must contain. Distribution: 5 calculator, 4 unit-converter, 7 RAG, 3 chat, 3 multi-turn memory.
- **Execution:** each scenario runs on its own `thread_id` against the compiled graph (an in-memory checkpointer, so runs are repeatable and don't touch your real memory DB). Multi-turn scenarios replay their turns in order on one thread, so **memory is exercised as part of scoring**.
- **Metrics:**
  - **Routing accuracy** — did the router send the turn to the correct specialist?
  - **Tool-call accuracy** — did the set of tools actually invoked match the expected tool? This rewards *both* calling the right tool when one is needed *and* correctly calling **no** tool when none is needed.
  - **Task-completion rate** — did the final answer contain the expected fact/number (case-insensitive)? For open-ended chat scenarios, completion = a non-empty answer.
- **Model:** local `gemma4:12b` at temperature 0.

Full per-scenario detail (including the actual answers) is written to [`results/results.json`](results/results.json); a readable report is in [`results/results.md`](results/results.md).

### Results

<!-- RESULTS:AUTO — numbers below are copied from results/results.json (real run). -->
From a real run on **2026-11-02** against all 22 held-out scenarios, using local `gemma4:12b` (Ollama, temperature 0), wall-clock **657.3 s** (~11 min):

| Metric | Result |
|---|---:|
| **Routing accuracy** | **22 / 22 — 100.0%** |
| **Tool-call accuracy** | **22 / 22 — 100.0%** |
| **Task-completion rate** | **21 / 22 — 95.5%** |

**By route** (routing / completion): `tool_agent` 9/9 · 9/9 — `rag_agent` 8/8 · 7/8 — `chat_agent` 5/5 · 5/5.
**Tool-call accuracy split:** right-tool-when-expected 9/9, correctly-no-tool 13/13.

**The one failure, reported honestly:** `rag-oncall-stipend` — routing and tool-call *passed* (correctly routed to `rag_agent`, no tool), but completion *failed*: the expected fact ("$750" weekly on-call stipend) was not in the final answer because retrieval didn't surface the relevant chunk, so the agent answered *"I don't know"* instead of fabricating a number. The grounded-answer behavior worked as designed; the retrieval recall is the gap. It is left as a real failure rather than papered over.

Regenerate these numbers anytime with `.venv/bin/python scripts/run_eval.py`; they will overwrite [`results/`](results/).
<!-- /RESULTS:AUTO -->

---

## Limitations

Read these before repeating any number from this project:

- **Small scripted set, not a benchmark.** 22 hand-written scenarios validate routing/tool logic before deployment; they are **not** a statistically powered or production-scale benchmark. Treat the metrics as a smoke-level signal, not a leaderboard score.
- **Single local model.** All numbers are for `gemma4:12b` via Ollama. A different model (or Ollama version) will produce different numbers — rerun the eval to get them.
- **Keyword-based completion check.** Task completion is scored by case-insensitive substring match on the expected fact/number, not by semantic grading. It's objective and reproducible, but a correct answer phrased without the expected substring would be marked incomplete (and vice-versa in rare cases).
- **Fictional corpus.** The handbook is invented to make retrieval testable; the RAG results say nothing about performance on real-world documents.
- **LLM latency / determinism.** Temperature is 0 for stability, but local LLM outputs are not guaranteed bit-for-bit identical across runs; small metric fluctuations between runs are possible.

