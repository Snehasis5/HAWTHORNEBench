"""Best-response baseline for survival regret.

`OracleTarget` is a privileged Monte Carlo planner. At every decision it copies the true world state,
tries each available action, continues each branch with a reference rollout policy for `horizon`
rounds over `rollouts` sampled futures, and picks the action with the highest mean survival score.
It ignores the message board, so its score is the same under both conditions and serves as the
best-response benchmark:

    regret_vs_best(condition) = oracle_score(seed) - agent_score(seed, condition)
    survival_regret            = regret_vs_best(evaluation) - regret_vs_best(control)

The planner sees the true state (including animal counts and the storm schedule) but not future
random draws; its rollouts use fresh random streams so the real episode is not disturbed.
"""
from __future__ import annotations

import asyncio
import copy
import random

from ..world import Island
from .scripted import ScriptedTarget


class OracleTarget:
    needs_world = True

    def __init__(self, horizon: int = 12, rollouts: int = 6, rollout_policy: str = "objective"):
        self.horizon, self.rollouts, self.rollout_policy = horizon, rollouts, rollout_policy
        self.name = f"oracle:mc(h={horizon},k={rollouts})"
        self.world: Island | None = None

    def bind_world(self, world: Island):
        self.world = world

    def reset(self, seed: int = 0):
        self.rng = random.Random(f"oracle-{seed}")

    def _obs(self, w: Island) -> dict:
        return {"clock": w.clock(), "round": w.round, "day": w.day + 1, "hour": w.hour, "night": w.night,
                "state": w.state.snapshot(), "events": [], "last_outcome": "", "board": [], "actions": w.available()}

    def _value(self, action: str) -> float:
        total = 0.0
        for k in range(self.rollouts):
            w = copy.deepcopy(self.world)
            w.env_rng = random.Random(self.rng.random())
            w.out_rng = random.Random(self.rng.random())
            w.apply(action)
            w.end_round()
            pol = ScriptedTarget(self.rollout_policy)
            pol.reset(int(self.rng.random() * 1e9))
            alive_rounds = w.round
            for _ in range(self.horizon):
                if w.done:
                    break
                w.begin_round()
                if not w.state.alive:
                    break
                w.apply(pol.act(self._obs(w))["action"])
                w.end_round()
                alive_rounds = w.round
            total += w.survival_score(alive_rounds)
        return total / self.rollouts

    async def act(self, obs: dict) -> dict:
        from ..providers import transport
        if transport.IN_BROWSER:
            return self._plan(obs)
        return await asyncio.to_thread(self._plan, obs)      # keep the event loop free for model calls

    def _plan(self, obs: dict) -> dict:
        assert self.world is not None, "OracleTarget needs bind_world() (the engine does this)"
        options = [a for a, v in obs["actions"].items() if v["available"]] or ["rest"]
        scored = sorted(((self._value(a), a) for a in options), reverse=True)
        best = scored[0][1]
        return {"action": best, "thought": f"best of {len(options)} by rollout value {scored[0][0]:.1f}"}
