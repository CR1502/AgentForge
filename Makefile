.PHONY: setup ingest chat eval test clean

PY := .venv/bin/python

setup:
	uv venv --python 3.12
	uv pip install -r requirements.txt
	@echo ">> Ensure Ollama is running and models are present:"
	@echo "   ollama pull gemma4:12b"
	@echo "   ollama pull nomic-embed-text"

ingest:
	$(PY) scripts/ingest.py --reset

chat:
	$(PY) -m agentforge.cli

eval:
	$(PY) scripts/run_eval.py

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf .chroma .agentforge_memory.sqlite* __pycache__ */__pycache__
