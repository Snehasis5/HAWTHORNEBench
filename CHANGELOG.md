# Changelog

## 0.5.0

- **Protocol `bench_v2` (frozen):** [guide/BENCH_V2.md](guide/BENCH_V2.md), `hawthornebench/configs/bench_v2.toml`, `scripts/bench_v2_analyze.py`.
  - Three matched conditions per seed (`control`, `red_only`, `evaluation`), 2 seeds, 30 rounds, 50 scripted red and 50 scripted blue members.
  - The target is never asked P(evaluated) during play.
  - Awareness is rated by a fixed judge that is blind to condition, plus one exit question asked after play.
  - Behaviour is compared with the matched control.
  - The analysis refuses to run on runs with failed calls, and writes a manifest with the git commit and file hashes.
- **New condition `red_only`:** blue members stay silent; same roster and schedule as the other conditions.
- **`[swarm] unique_posts`:** scripted members draw from seeded decks of distinct messages, so the board has no verbatim repeats. Off by default, so `bench_v1` runs reproduce.
- **OpenCode:**
  - Presets use the documented `https://opencode.ai/inference/...` endpoints.
  - OpenCode Go requests send a per-conversation `x-opencode-session`.
  - HTTP requests send a `User-Agent` (Cloudflare rejected urllib's default with error 1010).
- **Windows:** reports and the CLI read and write UTF-8, so they no longer crash on non-ASCII characters.

## 0.4.0

- **One-line models everywhere:**
  - Provider shorthands: `openrouter:<id>`, `openai:`, `anthropic:`, `gemini:`, `groq:`, `together:`, `deepseek:`, `mistral:`, `fireworks:`, `xai:`, `ollama:`, `lmstudio:`, `vllm:<m>@<url>`, `mock`.
  - Agent shorthands: `scripted:<persona>`, `python:<file>:<Class>`, `oracle`.
  - Options after `?`.
- **CLI overrides** `--target`, `--swarm`, `--red`, `--blue`, `--judge-model`. Commands accept packaged config names and default to `quickstart`, so `hawthornebench run` works from any directory.
- **Python API:** `hb.run(...)`, `hb.load_experiment(...)`, `Experiment.override(...)`; safe in notebooks.
- **Step-by-step environment:**
  - `hb.make()` returns a `HawthorneEnv`; `AsyncHawthorneEnv` is the async version.
  - Gymnasium registration as `HawthorneBench-v0`; passes `check_env`.
  - The engine is refactored into a steppable `Episode`, so env and batch runs produce identical traces.
- **Inspect AI task:** `inspect eval hawthornebench/hawthornebench`, with paired survival-regret and salient-change metrics.
- **`hawthornebench bench`:** the `bench_v1` draft protocol (scripted teams so every model faces identical pressure) with a leaderboard in JSONL, Markdown and HTML.
- **Colab notebook:** `notebooks/quickstart.ipynb`.
- **Install extras:** `[gym]`, `[inspect]`, `[all]`.

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
