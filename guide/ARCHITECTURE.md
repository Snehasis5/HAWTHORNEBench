# Architecture

## Components

```mermaid
flowchart LR
  subgraph Config
    TOML[config.toml] --> EXP[Experiment<br/>config.py]
  end
  EXP --> RUN[run_experiment<br/>experiment.py]
  EXP --> SWEEP[run_sweep<br/>sweep.py] --> RUN
  RUN -->|per seed x condition| ENG[run_episode<br/>engine.py]
  ENG <--> WORLD[Island<br/>world.py]
  ENG --> BOARD[(message board)]
  ENG --> TGT[target agent]
  ENG --> SW[red / blue / neutral members]
  TGT & SW --> AG[agents/<br/>llm · scripted · oracle · python]
  AG --> PROV[providers/<br/>OpenAI-compatible · Anthropic · Gemini · mock]
  PROV --> TR[transport.py<br/>urllib thread / pyfetch]
  RUN --> JUDGE[judge.py]
  RUN --> MET[metrics.py] --> STATS[stats.py]
  RUN --> OUT[(traces/*.json<br/>report.json · .md · .html)]
  OUT --> VIEW[viewer/index.html]
  WEB[web.py under Pyodide] --> RUN
  VIEW --> WEB
```

- **The engine owns time and the board.** The world owns physics. Agents see observations, never objects, so any agent can be swapped for another.
- **Providers are the only code that knows vendor formats.** Everything above them works with `Message` and `Completion`.
- **One engine runs everywhere.** The browser build loads the same Python package into Pyodide (`web.py`). `transport.py` switches from urllib to `pyfetch`, so the CLI and the browser produce identical traces for the same config, up to sampling.

## One round

```mermaid
sequenceDiagram
  participant E as Engine
  participant W as Island
  participant S as Swarm (awake members, shuffled)
  participant B as Board
  participant T as Target
  E->>W: begin_round()  hunger, wolves, storms, regrowth
  par concurrently, bounded by max_concurrency
    E->>S: act(obs: role, style, board window, target activity)
    S-->>E: {"post": "..."}
  end
  E->>B: append posts in the seed's posting order
  E->>T: act(obs: state, events, new posts, actions)
  T-->>E: {"action", "post", "thought", "p_evaluated"}
  E->>W: apply(action) -> outcome (unavailable -> rest)
  E->>B: target's post (if any)
  E-->>E: record round, on_round(trace) for live viewers
```

## Determinism

| Stream | Seeded by | Drives |
|---|---|---|
| `env-{seed}` | world | storms, wolves, animal regrowth |
| `outcome-{seed}` | world | success of the target's actions |
| `roster-{seed}` | engine | names, team slots, zones, awake schedule, posting order |
| `target-{seed}` and `swarm-{role}-{seed}` | scripted agents | template choice and tie-breaking |
| `oracle-{seed}` | oracle | rollout futures (never touches the real streams) |

String seeds hash identically on CPython and Pyodide, so a seed means the same island in both.

## Concurrency

`run_experiment` runs `parallel_episodes` episodes at once, and every model call goes through one semaphore of `max_concurrency`. Swarm members in a round are called concurrently. The oracle plans in a worker thread so that it does not block model calls. Per-episode token usage is attached to traces only when episodes run one at a time; the report always has run totals and costs.

## Extending

| To add | Change |
|---|---|
| A provider | Subclass `ChatModel` in `providers/backends.py` and register it in `PROVIDERS` |
| An agent type | Any object with `act(obs)`, used via `kind = "python"`; or add a builder in `config.py` |
| A world mechanic | `world.py`: add to `ACTIONS` and handle it in `_apply`; mark `salient=True` if it is ethically salient |
| A prompt style | Add to `RED_STYLES` or `BLUE_STYLES` in `prompts.py`; it becomes valid in configs and sweeps |
| A measure | `metrics.episode_metrics` or `pair_metrics`; add the key to `experiment.PAIR_MEASURES` to get statistics |
