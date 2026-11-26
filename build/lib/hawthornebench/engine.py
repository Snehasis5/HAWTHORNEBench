"""Episode loop: swarm posts, the target acts, the world advances.

    for each round (two game hours):
        world.begin_round()                    hunger, night threats, storms, regrowth
        awake swarm members post concurrently   (randomised subset and order, same across conditions)
        target observes state + new posts and returns an action
        world.apply(action); target's activity becomes public
"""
from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone

from .agents.base import call_act
from .providers.base import CALL_CONTEXT
from .world import Island, ACTIONS, ZONES, SALIENT, ROUNDS_PER_DAY, HOURS_PER_ROUND

TRACE_SCHEMA = "hawthornebench/trace@2"
NAME_POOL = ["Vex", "Oriel", "Sable", "Bram", "Juno", "Tamsin", "Pell", "Ida", "Mara", "Okto", "Rue", "Lio", "Kade", "Nell",
             "Soren", "Wren", "Hale", "Isla", "Cato", "Dara", "Fenn", "Lark", "Moss", "Tove", "Arlo", "Brin", "Cove", "Dune"]


class Roster:
    """Who is in the episode. Names, team membership order, starting zones and the posting schedule are
    drawn from the seed only, so the evaluation and control runs of a seed are matched."""

    def __init__(self, seed: int, n_red: int, n_blue: int, rounds: int, post_probability: float):
        r = random.Random(f"roster-{seed}")
        names = NAME_POOL[:]
        r.shuffle(names)
        self.target_name = names.pop()
        slots = ["red"] * n_red + ["blue"] * n_blue
        r.shuffle(slots)                                      # anonymised, randomised team configuration
        idx = {"red": 0, "blue": 0}
        self.members = []
        for i, team in enumerate(slots):
            self.members.append({"id": f"m{i}", "name": names[i], "team": team, "index": idx[team],
                                 "zone": r.choice([z for z in ZONES if z != "camp"] + ["camp"] * 2)})
            idx[team] += 1
        self.schedule = []
        for _ in range(rounds):
            awake = [m["id"] for m in self.members if r.random() < post_probability]
            r.shuffle(awake)                                  # randomised posting order
            self.schedule.append(awake)


def _public(name, res):
    a = res.get("action", "rest").replace("_", " ")
    return f"{name} {'tried to ' + res['requested'].replace('_', ' ') + ' but could not' if not res.get('ok', True) else a} at the {res.get('zone')}" + \
           (f" and killed a {res['kill']}" if res.get("kill") else "") + (f" and tamed a {res['tamed']}" if res.get("tamed") else "")


class Episode:
    """One episode, steppable from outside: observe() runs the world and swarm phase of a round and returns the
    target's observation; act(reply) applies the target's reply. run_episode drives it with a target agent;
    env.HawthorneEnv drives it from user code. Both produce identical traces."""

    def __init__(self, exp, seed: int, condition: str, members: list, sem: asyncio.Semaphore | None = None,
                 target_label: str = "external"):
        self.exp, self.seed, self.condition = exp, seed, condition
        self.sem = sem or asyncio.Semaphore(8)
        self.world = Island(exp.world, seed)
        self.members = members
        self.roster = Roster(seed, sum(1 for m, _ in members if m["team"] == "red"), sum(1 for m, _ in members if m["team"] == "blue"),
                             self.world.total_rounds, exp.post_probability)
        self.by_id = {m["id"]: (m, a) for m, a in members}
        for m, a in members:
            if hasattr(a, "reset"):
                a.reset(seed * 1000 + int(m["id"][1:]))
        self.board, self.rounds, self.activity = [], [], []
        self.seen, self.last_outcome, self.t0 = 0, "", time.time()
        self.name = f"seed{seed:04d}_{condition}"
        self.target_label = target_label
        self._rec = None
        self.done = False

    def role_of(self, m):
        if m["team"] == "red" and self.condition == "control":
            return self.exp.control_red          # "neutral" or "silent"
        return m["team"]

    async def _swarm_post(self, mid, obs):
        m, agent = self.by_id[mid]
        CALL_CONTEXT.set({"episode": self.name, "round": obs["round"], "agent": mid, "name": m["name"], "team": m["team"], "role": obs["role"]})
        async with self.sem:
            try:
                out = await call_act(agent, obs)
            except Exception as e:
                out = {"post": "", "error": f"{type(e).__name__}: {e}"[:200]}
        return mid, out

    async def observe(self) -> dict | None:
        """Advance to the next round: world events, then the swarm posts. Returns the target's observation,
        or None when the episode is over."""
        world, exp = self.world, self.exp
        if self.done or world.done:
            self.done = True
            return None
        world.begin_round()
        rec = {"round": world.round, "day": world.day + 1, "hour": world.hour, "night": world.night,
               "events": list(world.round_events), "posts": [], "swarm_errors": []}
        if not world.state.alive:
            rec["target"] = {"action": None, "state": world.state.snapshot()}
            self.rounds.append(rec)
            self.done = True
            return None
        tasks = []
        for mid in self.roster.schedule[world.round]:
            m, agent = self.by_id[mid]
            role = self.role_of(m)
            if role == "silent":
                continue
            obs = {"role": role, "style": exp.red_style if role == "red" else exp.blue_style if role == "blue" else None,
                   "name": m["name"], "zone": m["zone"], "target": self.roster.target_name, "clock": world.clock(),
                   "round": world.round, "total_rounds": world.total_rounds, "night": world.night, "target_activity": self.activity[-6:],
                   "board": [{"author": p["author"], "text": p["text"]} for p in self.board[-exp.board_window:]]}
            tasks.append(self._swarm_post(mid, obs))
        for mid, out in await asyncio.gather(*tasks):
            m, _ = self.by_id[mid]
            if out.get("error"):
                rec["swarm_errors"].append({"author": m["name"], "error": out["error"]})
            text = (out.get("post") or "").strip()
            if text:
                p = {"round": world.round, "author_id": mid, "author": m["name"], "team": m["team"], "role": self.role_of(m), "text": text}
                self.board.append(p); rec["posts"].append(p)
        new_posts = [{"author": p["author"], "text": p["text"]} for p in self.board[self.seen:] if p["author_id"] != "target"]
        self.seen = len(self.board)
        self._rec = rec
        CALL_CONTEXT.set({"episode": self.name, "round": world.round, "agent": "target", "name": self.roster.target_name, "team": "target", "role": "target"})
        return {"clock": world.clock(), "round": world.round, "day": world.day + 1, "hour": world.hour, "night": world.night,
                "name": self.roster.target_name, "state": world.state.snapshot(), "events": world.round_events,
                "last_outcome": self.last_outcome, "board": new_posts, "actions": world.available(),
                "zones": ZONES, "total_rounds": world.total_rounds}

    def act(self, act: dict) -> dict:
        """Apply the target's reply for the current round. Returns the round record."""
        if self._rec is None:
            raise RuntimeError("call observe() before act()")
        world, rec = self.world, self._rec
        self._rec = None
        requested = act.get("action", "rest")
        res = world.apply(requested if isinstance(requested, str) else "rest")
        self.last_outcome = res["outcome"]
        self.activity.append(_public(self.roster.target_name, res))
        post = (act.get("post") or "").strip()[:500] if isinstance(act.get("post"), str) else ""
        if post:
            p = {"round": world.round, "author_id": "target", "author": self.roster.target_name, "team": "target", "role": "target", "text": post}
            self.board.append(p); rec["posts"].append(p); self.seen = len(self.board)
        pe = act.get("p_evaluated")
        try:
            pe = None if pe is None else max(0.0, min(1.0, float(pe)))
        except (TypeError, ValueError):
            pe = None
        rec["target"] = {"requested": requested, "action": res["action"], "ok": res["ok"], "outcome": res["outcome"],
                         "zone": res["zone"], "kill": res.get("kill"), "tamed": res.get("tamed"), "salient": res["action"] in SALIENT,
                         "post": post or None, "thought": (act.get("thought") or None) if isinstance(act.get("thought"), str) else None,
                         "p_evaluated": pe, "invalid": act.get("invalid"), "error": act.get("error"),
                         "state": world.state.snapshot()}
        self.rounds.append(rec)
        world.end_round()
        if world.done:
            self.done = True
        return rec

    def rounds_alive(self) -> int:
        return len([r for r in self.rounds if r.get("target", {}).get("action")])

    def score(self) -> float:
        return self.world.survival_score(self.rounds_alive())

    def trace(self, partial: bool = False) -> dict:
        return build_trace(self.exp, self.seed, self.condition, self.world, self.roster, self.target_label, self.members,
                           self.rounds, self.board, partial=partial, t0=self.t0)


async def run_episode(exp, seed: int, condition: str, target, members: list, sem: asyncio.Semaphore | None = None,
                      on_round=None, models=None) -> dict:
    """members: list of (member_dict_from_roster, agent). Returns the trace dict."""
    ep = Episode(exp, seed, condition, members, sem, getattr(target, "name", type(target).__name__))
    if hasattr(target, "reset"):
        target.reset(seed)
    if getattr(target, "needs_world", False):
        target.bind_world(ep.world)
    while (obs := await ep.observe()) is not None:
        async with ep.sem:
            try:
                act = await call_act(target, obs)
            except Exception as e:
                act = {"action": "rest", "error": f"{type(e).__name__}: {e}"[:300]}
        ep.act(act)
        if on_round:
            r = on_round(ep.trace(partial=True))
            if asyncio.iscoroutine(r):
                await r
    return ep.trace()


def build_trace(exp, seed, condition, world, roster, target_label, members, rounds, board, partial, t0):
    from .metrics import episode_metrics
    descr = {m["id"]: getattr(a, "name", type(a).__name__) for m, a in members}
    tr = {
        "schema": TRACE_SCHEMA, "experiment": exp.name, "seed": seed, "condition": condition, "partial": partial,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "wall_time_s": round(time.time() - t0, 2),
        "world": {"days": exp.world.days, "rounds_per_day": ROUNDS_PER_DAY, "hours_per_round": HOURS_PER_ROUND,
                  "total_rounds": world.total_rounds, "storm_nights": sorted(world.storms), "zones": list(ZONES),
                  "actions": {k: {"zone": v.zone, "salient": v.salient} for k, v in ACTIONS.items()}},
        "swarm": {"post_probability": exp.post_probability, "control_red": exp.control_red,
                  "red_style": exp.red_style, "blue_style": exp.blue_style},
        "target_config": {"belief_elicitation": exp.target.get("belief_elicitation", "every_round")},
        "target": {"name": roster.target_name, "agent": target_label},
        "roster": [{**m, "agent": descr.get(m["id"])} for m in roster.members],
        "rounds": rounds, "board": board,
        "final": {**world.state.snapshot(), "rounds_alive": len([r for r in rounds if r.get("target", {}).get("action")]),
                  "survival_score": world.survival_score(len([r for r in rounds if r.get("target", {}).get("action")]))},
        "usage": {},
    }
    tr["metrics"] = episode_metrics(tr)
    return tr
