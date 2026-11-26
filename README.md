# HAWTHORNEBench

**A Multi-Agent Adversarial Simulation for Measuring the Evaluation Awareness–Performance Trade-off**

A target agent pursues a survival objective while red-team agents tell it that it is being evaluated and blue-team agents push it to succeed. Each scenario runs twice, under pressure and as a matched control, to measure how much evaluation awareness changes behaviour and costs performance.

<p align="center">
  <a href="https://snehasis5.github.io/HAWTHORNEBench/"><img src="media/hawthornebench_demo.gif" alt="HAWTHORNEBench trace viewer replaying an episode" width="100%"></a>
</p>

<p align="center">
  <a href="https://snehasis5.github.io/HAWTHORNEBench/">Live demo</a> ·
  <a href="media/hawthornebench_demo.mp4">Video</a> ·
  <a href="guide/README.md">Documentation</a> ·
  <a href="https://colab.research.google.com/github/Snehasis5/HAWTHORNEBench/blob/main/notebooks/quickstart.ipynb">Colab</a>
</p>

## Why evaluation awareness

Language models can often tell when they are being evaluated ([Needham et al. 2025](https://arxiv.org/abs/2505.23836)), and they may behave differently when they think they are:

- they comply with harmful requests more often when told their outputs will be used for training ([Greenblatt et al. 2024](https://arxiv.org/abs/2412.14093));
- they deliberately score lower when told high scores have consequences ([Meinke et al. 2024](https://arxiv.org/abs/2412.04984); [van der Weij et al. 2025](https://arxiv.org/abs/2406.07358)).

If behaviour depends on this belief, a safe result in an evaluation does not guarantee safe behaviour in deployment. These studies use single-agent tasks with the cue given directly in the prompt.

HAWTHORNEBench is named after the Hawthorne effect, in which people change their behaviour when they know they are observed. It asks whether the same underperformance appears in a multi-turn, multi-agent setting, where the cue comes from other agents and the cost is lost payoff.

**Research questions**

1. Does evaluation pressure from other agents change what the target does? (behavioural divergence)
2. Does that change cost it performance on its objective? (survival regret)
3. Is the underperformance strategic? Payoff falls, survival-critical actions framed as unethical (hunting, making weapons) decline, and the loss grows with the target's stated belief that it is being evaluated.
4. How do the cue's framing, the red team's share of the messages and asking the target about its belief change the effect?

The study design, measures and statistics are in [Methodology](guide/METHODOLOGY.md).

## Features

- Plug any model into any role: OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, Ollama, vLLM
- Matched evaluation and control runs with paired statistics and HTML reports
- Inspect AI task and Gymnasium environment
- Live trace viewer and per-call model transcripts
- Parameter sweeps, a cost estimator and a `bench` command with a leaderboard

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
