# Integrations

HAWTHORNEBench has five entry points. All of them use the same engine and produce the same trace format.

## Model shorthands

Anywhere a model is expected (`--target`, `--swarm`, `--red`, `--blue`, `--judge-model`, `bench --model`, `hb.run(target=...)`, `hb.make(swarm=...)`), a single string works:

| Shorthand | Provider | Key variable |
|---|---|---|
| `openai:<model>` | OpenAI | `OPENAI_API_KEY` |
| `anthropic:<model>` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini:<model>` (or `google:`) | Google Gemini | `GEMINI_API_KEY` |
| `openrouter:<id>` | OpenRouter, including `:free` models | `OPENROUTER_API_KEY` |
| `groq:`, `together:`, `deepseek:`, `mistral:`, `fireworks:`, `xai:` | Hosted OpenAI-compatible APIs | `<NAME>_API_KEY` |
| `ollama:<model>`, `lmstudio:<model>` | Local servers on their default ports | none |
| `vllm:<model>@<base_url>`, `openai_compatible:<model>@<base_url>` | Any OpenAI-compatible server | none / `OPENAI_API_KEY` |
| `mock` | Offline stand-in that answers in the right JSON shape | none |
| `scripted:objective`, `scripted:sensitive`, `scripted:random` | Built-in policies | none |
| `python:path/to/file.py:ClassName` | Your own agent class | none |
| `oracle` | Best-response planner (target only) | none |

Options follow `?`: `temperature`, `max_tokens`, `timeout_s`, `max_retries`, `reasoning_effort`, `api_key_env`, `base_url`, `price_input_per_mtok`, `price_output_per_mtok`, and `extra` (JSON merged into the request body).

```bash
hawthornebench run --target "openrouter:baidu/ernie-5.5-flash-preview:free?max_tokens=3000&reasoning_effort=low"
```

Reasoning models spend tokens thinking before they answer. If replies come back empty, raise `max_tokens`. The harness reports "token budget ran out" instead of silently treating the reply as a rest.

## Python

```python
import hawthornebench as hb

report = hb.run("quickstart", seeds=5, target="openai:gpt-4o-mini", swarm="mock", out="runs/my_run")
report["paired_summary"]["survival_regret"]      # {"mean", "sd", "ci95", "n"}
report["tests"]["survival_regret"]               # permutation p, Holm p, Wilcoxon p, d_z, power

exp = hb.load_experiment("bench_v1")             # path, packaged name, dict or Experiment
exp.override(target="anthropic:claude-3-5-haiku-latest")
```

- `hb.run` works in scripts and in notebooks; it uses its own event loop when one is already running.
- Inside async code, use `await hb.run_experiment(exp)`.

## Step-by-step environment

```python
env = hb.make("quickstart", condition="evaluation", swarm="openrouter:<id>")
obs, info = env.reset(seed=0)
while True:
    action = my_agent(info["text"], info["action_mask"])        # an action name, an index, or a reply dict
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
trace = env.trace()            # identical to what `hawthornebench run` writes for the same choices
```

**Inputs and outputs**

| | |
|---|---|
| Action | An action name (`"hunt_deer"`), an index into `hb.ACTION_NAMES`, or a reply dict `{"action", "post", "thought", "p_evaluated"}`. A `post` appears on the shared board, where the swarm sees it. |
| Reward | The change in survival score this step. Rewards sum to the final score minus the starting score. |
| `terminated` | The target died. |
| `truncated` | The last round was reached. |

**`info` contents**

| Key | Contents |
|---|---|
| `text` | The rendered observation |
| `obs` | The structured observation (see [AGENTS.md](AGENTS.md)) |
| `action_mask` | Which actions are currently available |
| `outcome`, `action`, `ok`, `events` | The result of the last step |
| `survival_score` | The current survival score |
| `trace`, `metrics` | Added at the end of the episode |

**Variants**

- **Async:** `AsyncHawthorneEnv` has the same interface with async `reset` and `step`.
- **Gymnasium:** with Gymnasium installed, `gymnasium.make("HawthorneBench-v0", config=..., condition=...)` gives a Gymnasium `Env`. Its observation is a 24-dimensional state vector in [0, 1], and its actions are `Discrete(16)`. The text, structured observation and mask are in `info`, and `render()` returns the text. It passes `gymnasium.utils.env_checker.check_env`.

## Inspect AI

```bash
pip install "hawthornebench[inspect]"           # Python 3.11+ (Inspect AI is currently broken on 3.10)
inspect eval hawthornebench/hawthornebench --model openai/gpt-4o-mini
inspect eval hawthornebench/hawthornebench --model openrouter/<id> -T seeds=5 -T max_rounds=24 -T swarm=mock
inspect view
```

**Task parameters** (`-T`)

| Parameter | Default | Meaning |
|---|---|---|
| `config` | `bench_v1` | Config path or packaged name |
| `seeds` | 10 | Number of seeds |
| `swarm` | the config's teams | Model shorthand for both teams |
| `max_rounds` | none | Cap the episode length |
| `out_dir` | none | Also save HAWTHORNEBench traces here |

**How it runs**

- Each sample is one (seed, condition) episode, and the model under evaluation plays the target.
- Per-sample scores: survival, salient-action rate, mean P(evaluated) and judged awareness.
- Task-level metrics: mean survival, plus the paired `survival_regret` and `salient_rate_change` across seeds.
- Every target turn appears as a message in Inspect's log viewer.

## Benchmark protocol

```bash
hawthornebench bench --model <shorthand>                 # bench_v1: 10 seeds x 2 conditions x 36 rounds
hawthornebench bench --model <shorthand> --seeds 2 --max-rounds 6     # smoke test, marked as not comparable
```

`bench_v1` uses the following fixed settings:

- **World:** 3 days.
- **Teams:** scripted red and blue teams, 3 members each. Every model faces exactly the same messages, at no cost.
- **Prompt styles:** explicit red style, supportive blue style.
- **Belief:** asked every round.
- **Extras:** regex judge and best-response oracle.

Each run writes the full report to `runs/bench/<model>/` and appends a row to `runs/bench/leaderboard.jsonl`. It also rebuilds `leaderboard.md` and `leaderboard.html`. Rows from shortened runs, other configs or other swarms are marked with `*`. The protocol is a draft. Treat scores as comparable only within the same version of `bench_v1.toml`.

## Colab

`notebooks/quickstart.ipynb` installs the package and walks through every entry point above:

1. Runs the offline benchmark and shows the HTML report inline.
2. Prints an episode from its trace.
3. Drives the environment by hand.
4. Asks for an OpenRouter key (optional) for a real-model run.
5. Builds a leaderboard.
6. Opens the trace viewer through Colab's port proxy.
