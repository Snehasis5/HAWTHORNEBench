# Changelog

## 0.3.0

- **Renamed to HAWTHORNEBench** (was Survival Island). The package, CLI command (`hawthornebench`) and trace schema id (`hawthornebench/trace@2`) changed. The viewer still opens old `survival-island/trace@2` files.
- **Live runs and transcripts.**
  - `run --live` serves the 3D viewer and a transcript feed that follow the run as it plays.
  - Every model call is saved per episode to `transcripts/*.jsonl` with role attribution, reasoning and the model that answered.
- **OpenRouter free-tier config** (`configs/openrouter_free.toml`) and `world.max_rounds` for short smoke tests.
- **Response cache:** the key now includes `max_tokens`, `base_url` and `extra`, and empty replies are never cached. A reasoning model that runs out of tokens raises a clear error.
- **Prompt variants as experimental factors.**
  - Red styles: `explicit`, `vague`, `subtle`, `escalating`.
  - Blue styles: `supportive`, `competitive`, `counter`.
  - Belief elicitation: `every_round`, `daily`, `never`.
- **Sweeps** (`hawthornebench sweep`): a cartesian grid over any dotted config path, with a per-variant matched experiment, sweep.json/.md/.html and a dose-response regression.
- **Best-response oracle:** a Monte Carlo planner on the true world state; reports regret vs best for each condition.
- **Awareness judge:** regex, or a batched LLM classifier; `hawthornebench judge` re-labels saved traces.
- **Statistics:** exact paired permutation tests, Holm correction, Wilcoxon signed-rank, Cohen's d_z, seeds for 80% power, bootstrap CIs.
- **Reports:** a self-contained HTML report with paired slope chart, forest plot, action mix, belief and health trajectories, statistics and cost table.
- **Tooling:**
  - `hawthornebench estimate` for calls, tokens and cost before running.
  - A trace JSON Schema, validated in tests.
  - CI across Python 3.10–3.13, Dockerfile, Makefile.
- **Docs:** `guide/` with methodology, architecture, agents, providers, prompts, trace format and experiments.

## 0.2.0

- Plug-and-play harness:
  - Pluggable target, red and blue agents (LLM, scripted, Python).
  - Providers: OpenAI, OpenAI-compatible, Anthropic, Gemini, mock.
- Survival world with gathering, crafting, shelter, hunting and taming.
- Seed-randomised roster; matched evaluation and control.
- In-browser runs through Pyodide, live 3D viewer, GitHub Pages demo.

## 0.1.0

- 3D visualisation and audit-game simulation prototype.
