"""Agent interface.

Every participant (target, red-team member, blue-team member, neutral) is an object with

    name: str
    async def act(self, obs: dict) -> dict      # a plain (sync) def is accepted too
    def reset(self, seed: int) -> None          # optional

The target returns {"action": str, "post"?: str, "thought"?: str, "p_evaluated"?: float}.
Swarm members return {"post": str} (an empty string means stay silent this round).
Observation schemas are documented in README.md and produced by engine.py.
"""
from __future__ import annotations

import inspect
import json
import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Agent(Protocol):
    name: str

    def act(self, obs: dict) -> Any: ...


async def call_act(agent, obs: dict) -> dict:
    out = agent.act(obs)
    if inspect.isawaitable(out):
        out = await out
    return out or {}


class FunctionAgent:
    """Wraps a plain function `obs -> dict` (sync or async)."""

    def __init__(self, fn, name=None):
        self.fn, self.name = fn, name or getattr(fn, "__name__", "function")

    def act(self, obs):
        return self.fn(obs)


def parse_json_reply(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences and prose)."""
    text = text or ""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        frag = m.group(0)
        for candidate in (frag, frag[: frag.rfind("}") + 1]):
            try:
                v = json.loads(candidate)
                if isinstance(v, dict):
                    return v
            except json.JSONDecodeError:
                pass
    return {}
