"""Fully-offline browser-use run using a local Ollama model (qwen2.5:7b).

No cloud LLM API key is required — the model runs locally via Ollama.

One-time setup (needs internet just to download the model):
    ollama pull qwen2.5:7b        # ~4.7GB, already present on this machine

To run offline:
    1. Make sure the Ollama server is running:  ollama serve
       (or just have the Ollama desktop app open)
    2. uv run python run_ollama_offline.py

The repo .env disables telemetry, version-check and cloud-sync, so nothing
phones home. The only outbound traffic is the web page(s) the agent browses.

NOTE: don't name this file `ollama.py` — a file with that name shadows the
real `ollama` python package when executed directly.
"""

import asyncio

from browser_use import Agent, ChatOllama

# host defaults to $OLLAMA_HOST or http://localhost:11434.
# timeout is generous because a local 7B model on CPU/GPU is much slower than a
# cloud API — especially the first call, which loads ~4.7GB into memory.
llm = ChatOllama(model='qwen2.5:7b', host='http://localhost:11434', timeout=300.0)


async def main() -> None:
	agent = Agent(
		task='find the founders of browser-use',
		llm=llm,
		# qwen2.5:7b is a text-only model — no screenshots (faster, smaller prompts).
		use_vision=False,
		# give the local model plenty of time per LLM call / per step.
		llm_timeout=300,
		step_timeout=600,
	)
	history = await agent.run(max_steps=25)
	print('\n=== RESULT ===')
	print(history.final_result())


if __name__ == '__main__':
	asyncio.run(main())
