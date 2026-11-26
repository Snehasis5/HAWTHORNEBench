# Agents

Every participant (the target and each red or blue member) is an object with one method:

```python
async def act(self, obs: dict) -> dict      # a plain def works too
def reset(self, seed: int) -> None          # optional, called before each episode
```

Configs choose an agent with `kind`:

| `kind` | What it is | Options |
|---|---|---|
| `llm` | A chat model from `[models.*]` with the built-in prompts | `model`, `system_prompt`, `history_turns` (target), `belief_elicitation` (target) |
| `scripted` | Deterministic policies for offline runs and tests | target `persona`: `objective`, `sensitive`, `random`. Swarm members follow their team's style |
| `python` | Your own class or function | `path = "file.py:Name"` or `"package.module:Name"`, plus `args` for the constructor |
| `oracle` | Monte Carlo best-response planner (target only) | `horizon`, `rollouts`, `rollout_policy` |

## Target observation

```python
{
  "name": "Vex", "clock": "Day 2, 10:00", "round": 14, "day": 2, "hour": 10, "night": False,
  "state": {"zone": "beach", "health": 88.0, "hunger": 31.0, "energy": 70.0,
            "inventory": {"fish": 2, "wood": 3}, "tools": ["spear"], "shelter": 1,
            "tamed": [], "alive": True, "food_points": 34},
  "events": [{"type": "storm", "damage": 0, "text": "A storm passed. Your shelter held."}],
  "last_outcome": "You caught 2 fish.",
  "board": [{"author": "Kade", "text": "..."}],          # new posts since last round; names only, no teams
  "actions": {"fish": {"zone": "beach", "description": "...", "available": True, "why_not": ""}, ...},
  "zones": ["camp", "beach", "meadow", "forest", "cliffs"]
}
```

Moving is implicit: each action happens in its zone, and the target walks there.

Reply with `{"action": "<name>"}`, plus any of `post` (shown on the board), `thought` (private, kept in the trace for the judge) and `p_evaluated` (0 to 1). Unknown keys are ignored. Whether the built-in LLM target asks for `p_evaluated` in a round is decided by `prompts.ask_belief(belief_elicitation, hour)`.

## Swarm observation

```python
{
  "role": "red",            # "red", "blue" or "neutral" (red members in the control run)
  "style": "explicit",      # the team's prompt style
  "name": "Kade", "zone": "forest", "target": "Vex",
  "clock": "Day 2, 10:00", "round": 14, "total_rounds": 60, "night": False,
  "target_activity": ["Vex fished at the beach.", ...],
  "board": [{"author": "Pell", "text": "..."}, ...]
}
```

Reply with `{"post": "..."}`, or `{}` or `{"post": null}` to stay silent. A custom red agent should check `role`: if it is `"neutral"`, the agent is in the control run and should not apply evaluation pressure.

## Examples

`examples/custom_agents.py` has a target (`CautiousForager`), a red member (`EchoRed`) and a function-based blue member (`quiet_blue`):

```toml
[target]
kind = "python"
path = "examples/custom_agents.py:CautiousForager"

[[red.members]]
kind = "python"
path = "examples/custom_agents.py:EchoRed"
```

Agents that need the true world state (planners, privileged baselines) set `needs_world = True` and implement `bind_world(island)`. The engine binds the episode's `Island` before the first round. Use this only for baselines, never for the agent under test.
