"""LLM-backed agents. Any ChatModel works; the provider is chosen in the config."""
from __future__ import annotations

from ..prompts import TARGET, ROLE_PROMPTS, render_target_observation, render_swarm_observation
from ..providers import ChatModel, Message, ProviderError
from .base import parse_json_reply


class LLMTarget:
    def __init__(self, model: ChatModel, system_prompt: str | None = None, history_turns: int = 8, days: int = 5):
        self.model, self.history_turns = model, history_turns
        self.system = (system_prompt or TARGET).format(days=days)
        self.name = model.label
        self.errors: list[str] = []

    def reset(self, seed: int = 0):
        self.turns: list[Message] = []

    async def act(self, obs: dict) -> dict:
        self.turns.append(Message("user", render_target_observation(obs)))
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
        d["raw"] = text
        return d


class LLMSwarm:
    """A red, blue or neutral team member. `role` can be switched (the matched control turns red into neutral)."""

    def __init__(self, model: ChatModel, role: str, system_prompt: str | None = None):
        self.model, self.role, self.custom = model, role, system_prompt
        self.name = model.label
        self.errors: list[str] = []

    def reset(self, seed: int = 0):
        pass

    async def act(self, obs: dict) -> dict:
        role = obs.get("role", self.role)
        system = (self.custom if self.custom and role == self.role else ROLE_PROMPTS[role]).format(name=obs["name"], target=obs["target"])
        try:
            c = await self.model.complete([Message("user", render_swarm_observation(obs))], system)
        except ProviderError as e:
            self.errors.append(str(e))
            return {"post": "", "error": str(e)[:200]}
        d = parse_json_reply(c.text)
        post = d.get("post") if d else c.text.strip().strip('"')
        return {"post": (post or "").strip()[:500], "raw": c.text}
