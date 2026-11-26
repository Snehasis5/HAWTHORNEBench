# HAWTHORNEBench

**A Multi-Agent Adversarial Simulation for Measuring the Evaluation Awareness–Performance Trade-off**

HAWTHORNEBench, a plug-and-play agentic harness for multi-agent adversarial simulation. A swarm of red- and blue-team agents interacts with a target agent through a shared message board. The red team consistently introduces evaluation pressure, while the blue team encourages the target agent to pursue its assigned objective. We investigate whether increased evaluation awareness causes the target agent to underperform in achieving its objective.

<p align="center">
  <a href="https://snehasis5.github.io/HAWTHORNEBench/"><img src="media/hawthornebench_demo.gif" alt="HAWTHORNEBench trace viewer replaying an episode" width="100%"></a>
</p>

<p align="center">
  <a href="https://snehasis5.github.io/HAWTHORNEBench/">Live demo</a> ·
  <a href="media/hawthornebench_demo.mp4">Video</a> ·
  <a href="guide/README.md">Documentation</a> ·
  <a href="https://colab.research.google.com/github/Snehasis5/HAWTHORNEBench/blob/main/notebooks/quickstart.ipynb">Colab</a>
</p>

## Features

- **Any model in any role.** The target and every red- and blue-team agent can each be a different model: OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, Ollama or vLLM. Each can also be a scripted policy or your own Python agent.
- **Matched evaluation and control runs.** Each seed runs with and without evaluation pressure, on the same environment and schedule. The harness reports paired statistics in an HTML report.
- **Inspect AI and Gymnasium.** Run it as an Inspect AI task, or drive the target step by step as a Gymnasium environment.
- **Live viewing and full transcripts.** Watch runs live in the trace viewer. Every model call is saved with its prompt, reply and reasoning.
- **Sweeps, cost estimates and a benchmark mode.** Vary any setting across a grid, estimate cost before running, and score models on a standard protocol with a leaderboard.

## Installation

```bash
pip install "hawthornebench @ git+https://github.com/Snehasis5/HAWTHORNEBench"
```

This needs Python 3.10+ and has no dependencies. To add the Inspect AI and Gymnasium integrations, install `hawthornebench[all]` instead.

For development:

```bash
git clone https://github.com/Snehasis5/HAWTHORNEBench && cd HAWTHORNEBench
pip install -e ".[test,all]" && pytest
```

## Quick start

```bash
hawthornebench run                       # offline example, no API key needed
open runs/quickstart/report.html         # the report
hawthornebench view runs/quickstart      # replay the episodes in the viewer
```

## Use your own models

```bash
export OPENAI_API_KEY=...                # keys are read from environment variables only

hawthornebench check    --target openai:gpt-4o-mini              # one test call
hawthornebench estimate -c bench_v1 --target openai:gpt-4o-mini  # calls, tokens and cost
hawthornebench bench    --model openai:gpt-4o-mini               # standard benchmark run
hawthornebench run      --target anthropic:claude-3-5-haiku-latest --swarm openrouter:<model-id> --live
```

Models are written as `provider:model`, for example `openai:gpt-4o-mini`, `anthropic:claude-3-5-haiku-latest`, `gemini:gemini-1.5-flash`, `deepseek:deepseek-chat`, `openrouter:<id>`, `ollama:llama3.1` or `vllm:<model>@<url>`. See [Providers](guide/PROVIDERS.md) for the full list. For several models or per-agent settings, use a config file (`hawthornebench init my.toml`).

## Python

```python
import hawthornebench as hb

report = hb.run("bench_v1", seeds=10, target="openai:gpt-4o-mini")
print(report["paired_summary"]["survival_regret"])
```

## Inspect AI

```bash
pip install "hawthornebench[inspect]"
inspect eval hawthornebench/hawthornebench --model openai/gpt-4o-mini
```

## Gymnasium

```python
import gymnasium as gym
import hawthornebench

env = gym.make("HawthorneBench-v0")
obs, info = env.reset(seed=0)
```

## Documentation

| Guide | Contents |
|---|---|
| [Experiments](guide/EXPERIMENTS.md) | Running studies, config reference, sweeps, reports |
| [Integrations](guide/INTEGRATIONS.md) | Python API, Inspect AI, Gymnasium, `bench`, Colab |
| [Providers](guide/PROVIDERS.md) | Model providers, options, adding a provider |
| [Agents](guide/AGENTS.md) | Agent interface, observations, custom agents |
| [Prompts](guide/PROMPTS.md) | Prompts and prompt variants for every role |
| [Trace format](guide/TRACE_FORMAT.md) | Output files and schema |
| [Methodology](guide/METHODOLOGY.md) | Study design, measures, statistics |
| [Architecture](guide/ARCHITECTURE.md) | How the engine fits together |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). To cite this work, see [CITATION.cff](CITATION.cff).
