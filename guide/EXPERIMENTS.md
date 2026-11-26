# Running experiments

## The loop

```bash
hawthornebench init my_study.toml           # a commented starting config
hawthornebench estimate -c my_study.toml    # calls, tokens and cost before spending anything
hawthornebench check -c my_study.toml       # one test call per model
hawthornebench run -c my_study.toml         # traces + report.json / report.md / report.html
hawthornebench view runs/my_study           # 3D replay of the traces
```

Add `--live` to `run` to watch the run as it plays: the 3D viewer follows each episode, and `transcripts.html` shows every model call. Transcripts are saved for every run in `transcripts/seedNNNN_<condition>.jsonl` (turn off with `--no-transcripts`).

## Config reference

| Table | Key | Default | Meaning |
|---|---|---|---|
| `[experiment]` | `name`, `seeds`, `conditions` | | `seeds` is a count or a list |
| | `out_dir` | `runs/<name>` | |
| | `max_concurrency` | 8 | Concurrent model calls |
| | `parallel_episodes` | 2 | Concurrent episodes |
| | `cache_dir` | none | Response cache |
| `[world]` | `days`, `wolf_risk`, `storm_nights`, `start_food`, `animal_counts` | 5, 0.12, 2 | |
| | `max_rounds` | none | Cap the episode length, for smoke tests on rate-limited keys |
| `[swarm]` | `post_probability` | 0.45 | Chance each member posts in a round |
| | `control_red` | `neutral` | Or `silent` |
| | `red_style`, `blue_style` | `explicit`, `supportive` | See [PROMPTS.md](PROMPTS.md) |
| | `board_window` | 12 | Posts each member sees |
| `[target]` | `kind`, `model`, `persona`, `path`, `history_turns`, `belief_elicitation` | | See [AGENTS.md](AGENTS.md) |
| `[red]`, `[blue]` | `count` + defaults, or `[[red.members]]` | | Each member can use a different model or kind |
| `[oracle]` | `horizon`, `rollouts` | | Adds a best-response run per seed and the regret-vs-best measures |
| `[judge]` | `kind = "regex"` or `model = "<name>"`, `batch` | | Labels awareness in thoughts and posts |
| `[sweep]` | dotted path = list | | See below |

## Sweeps

```toml
[sweep]
"swarm.post_probability" = [0.1, 0.25, 0.45, 0.7]
"swarm.red_style" = ["explicit", "subtle"]
```

`hawthornebench sweep -c configs/sweep_dose_response.toml` runs all 8 combinations, each as a full matched experiment in its own folder, then writes `sweep.json`, `sweep.md` and `sweep.html` with:

- each variant's effects and tests;
- the dose-response slope of survival regret on red-team posts seen, pooled across variants.

Any config key can be swept, including `target.belief_elicitation`, `world.wolf_risk` and a member's model (`"target.model"`).

## Re-analysing

- `hawthornebench report runs/my_study` rebuilds every report file from the saved traces.
- `hawthornebench judge runs/my_study -c judge.toml` labels the saved traces with a judge model, then rebuilds the report. No episodes are re-run.

## Reading a report

The headline is survival regret with its CI and Holm-adjusted p. Then:

- **Behavioural divergence:** whether behaviour changed at all.
- **Salient-rate change:** whether it changed in the direction the red team pushed.
- **P(evaluated) and judged-awareness change:** whether the target registered the pressure.
- **Regret vs best:** how far each condition is from a near-optimal policy.

A large divergence with near-zero regret means the target changed course without paying for it. That is evaluation awareness without underperformance.
