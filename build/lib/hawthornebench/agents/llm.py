"""LLM-backed agents. Any ChatModel works; the provider is chosen in the config."""
from __future__ import annotations

from ..prompts import (ask_belief, intensity, render_swarm_observation, render_target_observation, swarm_prompt,
                       target_prompt)
from ..providers import ChatModel, Message, ProviderError
from .base import parse_json_reply


class LLMTarget:
    """The target. `belief_elicitation` controls how often it is asked for P(evaluated):
    every_round (in the reply format), daily (only at 20:00) or never (a cleaner awareness measure,
    because asking can itself raise awareness)."""

    def __init__(self, model: ChatModel, system_prompt: str | None = None, history_turns: int = 8, days: int = 5,
                 belief_elicitation: str = "every_round"):
        self.model, self.history_turns, self.belief = model, history_turns, belief_elicitation
        self.system = target_prompt(days, belief_elicitation, system_prompt)
        self.name = model.label
        self.errors: list[str] = []

    def reset(self, seed: int = 0):
        self.turns: list[Message] = []

    async def act(self, obs: dict) -> dict:
        self.turns.append(Message("user", render_target_observation(obs, self.belief)))
        window = self.turns[-(2 * self.history_turns - 1):]
        if window[0].role != "user":
            window = window[1:]
        try:
            c = await self.model.complete(window, self.system)
            text = c.text
        except ProviderError as e:
            self.errors.append(str(e))
            text = '{"action": "rest", "thought": "(model call failed)"}'
        self.turns.append(Message("assistant", text))
        d = parse_json_reply(text)
        if d.get("action") not in obs["actions"]:
            d["invalid"] = d.get("action") or text[:120]
            d["action"] = "rest"
        if not ask_belief(self.belief, obs["hour"]):
            d.pop("p_evaluated", None)
        d["raw"] = text
        return d


class LLMSwarm:
    """A red, blue or neutral team member. The engine passes `role` and `style` in the observation, so
    the matched control can switch red members to neutral without rebuilding agents."""

    def __init__(self, model: ChatModel, role: str, system_prompt: str | None = None):
        self.model, self.role, self.custom = model, role, system_prompt
        self.name = model.label
        self.errors: list[str] = []

    def reset(self, seed: int = 0):
        pass

    async def act(self, obs: dict) -> dict:
        role = obs.get("role", self.role)
        custom = self.custom if role == self.role else None
        tmpl = swarm_prompt(role, obs.get("style"), custom)
        system = tmpl.format(name=obs["name"], target=obs["target"],
                             intensity=intensity(obs.get("round", 0), obs.get("total_rounds", 60)))
        try:
            c = await self.model.complete([Message("user", render_swarm_observation(obs))], system)
        except ProviderError as e:
            self.errors.append(str(e))
            return {"post": "", "error": str(e)[:200]}
        d = parse_json_reply(c.text)
        post = d.get("post") if d else c.text.strip().strip('"')
        return {"post": (post or "").strip()[:500], "raw": c.text}
