# AgentForge

AgentForge is a multi-agent assistant built on [LangGraph](https://langchain-ai.github.io/langgraph/). A single graph takes a user message, works out what kind of request it is, and routes it to one of three specialist agents: one that calls tools (a calculator and a unit converter), one that answers from a document corpus using retrieval (RAG), and one that just holds a conversation. The graph keeps memory across turns, and the repo ships with a test harness that measures how well the routing and tool use actually behave.

Everything runs locally through [Ollama](https://ollama.com/), so there are no API keys and no external calls. The evaluation can be reproduced on any machine with the same setup.

## How it works

Every turn starts at a router. The router classifies the message and writes a route into the graph state but produces no answer itself. Conditional edges then send the turn to the matching agent:

| Route | Agent | Behavior |
|---|---|---|
| `tool_agent` | Tool-calling agent | Calls real LangChain tools: `calculator` (AST-based arithmetic, no `eval`) or `unit_converter` (length, mass, temperature). |
| `rag_agent` | Retrieval agent | Retrieves from a persistent Chroma vector store over the document corpus and answers using only the retrieved text. |
| `chat_agent` | Conversational agent | Answers directly from the conversation so far. No tools, no retrieval. |

Memory is handled by a LangGraph `SqliteSaver` checkpointer keyed on a thread id, so a conversation keeps its context across turns and across restarts of the process.

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

   state: messages (add_messages reducer), route, retrieved_context, tool_calls_made
   memory: SqliteSaver checkpointer, scoped per thread_id (persists across runs)
```

### Components

- Orchestration: LangGraph `StateGraph` and `ToolNode`, with a `SqliteSaver` checkpointer.
- Model: `gemma4:12b` served by Ollama, with native tool-calling, run at temperature 0.
- Embeddings: `nomic-embed-text` served by Ollama.
- Vector store: a persistent Chroma collection built from the markdown corpus.
- Tools: two LangChain tools, `calculator` and `unit_converter`.

### Project layout

```
agentforge/
  state.py      # AgentState schema plus the routing/tool taxonomy
  config.py     # models, paths, and other settings
  tools.py      # calculator + unit_converter (LangChain tools)
  rag.py        # Chroma vector store: build and retrieve
  graph.py      # the LangGraph graph: router, specialists, memory
  cli.py        # interactive multi-turn chat
data/corpus/    # the document corpus (a fictional "Helix Dynamics" handbook)
eval/
  scenarios.json  # the 22 test scenarios (expected route/tool/answer)
  evaluate.py     # the evaluation engine and metrics
scripts/
  ingest.py     # build the vector store
  run_eval.py   # run the full evaluation and write to results/
results/        # results.json and results.md from the latest run
tests/          # structural smoke tests (no model required)
docs/           # architecture notes
```

The corpus is a made-up company handbook for a fictional company called Helix Dynamics. Using invented facts is deliberate: if the model answers a handbook question correctly, it had to get the answer from retrieval rather than from anything it saw during training.

## Setup

### Requirements

- [Ollama](https://ollama.com/), installed and running
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.12 (uv will provision it)

### Install

```bash
# 1. Clone
git clone https://github.com/CR1502/AgentForge.git
cd AgentForge

# 2. Pull the local models (chat and embeddings)
ollama pull gemma4:12b
ollama pull nomic-embed-text

# 3. Create the environment and install dependencies
uv venv --python 3.12
uv pip install -r requirements.txt

# 4. Build the vector store from the corpus
.venv/bin/python scripts/ingest.py --reset
```

The `Makefile` wraps the common commands: `make setup`, `make ingest`, `make chat`, `make eval`, and `make test`.

Dependencies are pinned in [`requirements.txt`](requirements.txt) to the versions used to produce the results in this repo (LangGraph and LangChain 1.x). The full resolved set is in [`requirements.lock.txt`](requirements.lock.txt).

## Running it

Interactive chat:

```bash
.venv/bin/python -m agentforge.cli
# it prints a thread id; resume that conversation later with:
.venv/bin/python -m agentforge.cli --thread <that-id>
```

Each reply also prints the route it took and any tools it called.

Run the evaluation (this regenerates `results/`):

```bash
.venv/bin/python scripts/run_eval.py
```

Smoke tests (fast, no model needed):

```bash
.venv/bin/python -m pytest tests/ -q
```

## Evaluation

The repo includes a fixed set of 22 test scenarios in [`eval/scenarios.json`](eval/scenarios.json). Each scenario lists the message or messages to send, the route it should take, the tool it should or should not call, and text the final answer should contain. The set breaks down as 5 calculator, 4 unit conversion, 7 retrieval, 3 plain chat, and 3 multi-turn memory cases.

The harness runs every scenario on its own thread against the compiled graph, using an in-memory checkpointer so the runs don't touch the on-disk memory database. Multi-turn scenarios replay their turns in order on a single thread, so memory is part of what gets tested.

Three things are measured:

- Routing accuracy: whether the router picked the right agent.
- Tool-call accuracy: whether the tools actually called match what the scenario expected, including the scenarios where no tool should be called.
- Task-completion rate: whether the final answer contains the expected fact or number, matched case-insensitively. For open-ended chat scenarios, any non-empty answer counts.

Full per-scenario output is written to [`results/results.json`](results/results.json), with a readable summary in [`results/results.md`](results/results.md).

### Results

Latest run on 2026-11-02, 22 scenarios, `gemma4:12b` at temperature 0, about 11 minutes of wall-clock time:

| Metric | Result |
|---|---:|
| Routing accuracy | 22 / 22 (100.0%) |
| Tool-call accuracy | 22 / 22 (100.0%) |
| Task-completion rate | 21 / 22 (95.5%) |

By route, routing and completion were: `tool_agent` 9/9 and 9/9, `rag_agent` 8/8 and 7/8, `chat_agent` 5/5 and 5/5. Tool-call accuracy was 9/9 where a tool was expected and 13/13 where none was.

One scenario failed: `rag-oncall-stipend`. It routed correctly to the retrieval agent and called no tool, but the answer did not contain the expected figure. Retrieval did not surface the chunk that held the on-call stipend, so the agent said it did not know instead of guessing. This is a retrieval recall gap, and it is left in the results as it happened.

To regenerate the numbers, run `.venv/bin/python scripts/run_eval.py`; it overwrites `results/`.

## Scope and limitations

- The test set is 22 hand-written scenarios. That is enough to check that routing and tool use behave as intended, but it is not a large or statistically rigorous benchmark.
- All numbers come from one local model, `gemma4:12b`. A different model, or a different Ollama version, will give different results; rerun the evaluation to see them.
- Completion is judged by substring match, not semantic grading. It is simple and repeatable, but a correct answer worded differently from the expected text would be counted as incomplete.
- The corpus is fictional, so the retrieval results say nothing about performance on real documents.
- Generation runs at temperature 0, but local model output is not guaranteed to be identical on every run, so small differences between runs are possible.
