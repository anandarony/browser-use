# Fully-offline local run using Ollama — no cloud API key required.
#
# One-time setup:
#   1. Install Ollama: https://github.com/ollama/ollama
#   2. Pull the model (once, needs internet): `ollama pull qwen2.5:7b`  (~4.7GB)
#
# To run offline:
#   1. Start the local server: `ollama serve`  (or just have the Ollama app running)
#   2. Run this script: `uv run python examples/models/ollama.py`
#
# The .env in the repo root already disables telemetry, version-check and cloud-sync,
# so nothing phones home — the only network traffic is the web page the agent browses.

from browser_use import Agent, ChatOllama

# host defaults to $OLLAMA_HOST or http://localhost:11434
llm = ChatOllama(model='qwen2.5:7b', host='http://localhost:11434')

Agent('find the founders of browser-use', llm=llm).run_sync()
