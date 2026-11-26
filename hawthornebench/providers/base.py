"""Provider-agnostic chat interface.

Every backend implements `ChatModel.complete(messages, system=...) -> Completion`. Agents never
talk to a vendor SDK directly, so the target, the red team and the blue team can each run on a
different provider. All HTTP goes through `transport.post_json`, which uses urllib in a thread on
CPython and `pyodide.http.pyfetch` in the browser.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import transport


@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    errors: int = 0
    cached: int = 0
    latency_s: float = 0.0

    def add(self, other: "Usage"):
        for k in ("input_tokens", "output_tokens", "calls", "errors", "cached"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.latency_s += other.latency_s

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    raw: Optional[dict] = None
    reasoning: Optional[str] = None      # chain of thought, when the provider returns it
    served_by: Optional[str] = None      # the model that actually answered (routers, fallbacks)


class ProviderError(RuntimeError):
    pass


@dataclass
class ModelSpec:
    """One entry of [models.<name>] in a config file."""
    name: str
    provider: str                      # openai | openai_compatible | anthropic | gemini | mock
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    api_key: str = ""                  # discouraged in files; used by the browser runner
    temperature: float = 0.7
    max_tokens: int = 400
    timeout_s: float = 60.0
    max_retries: int = 4
    extra: dict = field(default_factory=dict)   # merged into the request body (e.g. top_p, seed)
    headers: dict = field(default_factory=dict)
    price_input_per_mtok: float = 0.0          # USD per million input tokens, for cost reports
    price_output_per_mtok: float = 0.0

    def cost(self, usage: "Usage") -> float:
        return round(usage.input_tokens / 1e6 * self.price_input_per_mtok + usage.output_tokens / 1e6 * self.price_output_per_mtok, 4)

    def key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            v = os.environ.get(self.api_key_env, "")
            if not v and self.provider not in ("mock",) and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
                raise ProviderError(f"model '{self.name}': environment variable {self.api_key_env} is not set")
            return v
        return ""


# ------------------------------------------------------------------ transcripts
# Every model call is appended to <TRANSCRIPT_DIR>/<episode>.jsonl when a transcript directory is set.
# The engine fills CALL_CONTEXT (episode, round, agent, role) before each agent call; asyncio tasks
# inherit it, so concurrent swarm calls are attributed correctly.
TRANSCRIPT_DIR: contextvars.ContextVar = contextvars.ContextVar("transcript_dir", default=None)
CALL_CONTEXT: contextvars.ContextVar = contextvars.ContextVar("call_context", default=None)


def log_call(model, messages, system, completion, error, t0):
    d = TRANSCRIPT_DIR.get()
    if not d:
        return
    ctx = CALL_CONTEXT.get() or {}
    rec = {"ts": round(t0, 3), "latency_s": round(time.time() - t0, 3), "model": model.label, **ctx,
           "system": system, "messages": [m.__dict__ for m in messages],
           "response": completion.text if completion else None, "error": error,
           "reasoning": getattr(completion, "reasoning", None), "served_by": getattr(completion, "served_by", None),
           "usage": completion.usage.to_dict() if completion else None}
    p = Path(d) / f"{ctx.get('episode', 'calls')}.jsonl"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


class ChatModel:
    """Base class with retries, usage accounting and an optional on-disk response cache."""

    def __init__(self, spec: ModelSpec, cache_dir: str | Path | None = None):
        self.spec = spec
        self.usage = Usage()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._lock = asyncio.Lock()

    @property
    def label(self) -> str:
        return f"{self.spec.provider}:{self.spec.model or self.spec.name}"

    async def _request(self, messages: list[Message], system: str) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError

    def _cache_path(self, messages, system):
        h = hashlib.sha256(json.dumps([self.spec.provider, self.spec.model, self.spec.base_url, self.spec.temperature,
                                       self.spec.max_tokens, self.spec.extra, system,
                                       [m.__dict__ for m in messages]], sort_keys=True, default=str).encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    async def complete(self, messages: list[Message], system: str = "") -> Completion:
        t0 = time.time()
        try:
            c = await self._complete(messages, system)
        except ProviderError as e:
            log_call(self, messages, system, None, str(e), t0)
            raise
        log_call(self, messages, system, c, None, t0)
        return c

    async def _complete(self, messages: list[Message], system: str = "") -> Completion:
        if self.cache_dir:
            p = self._cache_path(messages, system)
            if p.exists():
                c = Completion(json.loads(p.read_text())["text"], Usage(cached=1), served_by="cache")
                self.usage.add(c.usage)
                return c
        delay = 1.0
        last = None
        for attempt in range(self.spec.max_retries + 1):
            t0 = time.perf_counter()
            try:
                c = await self._request(messages, system)
                c.usage.calls += 1
                c.usage.latency_s += time.perf_counter() - t0
                self.usage.add(c.usage)
                if self.cache_dir and c.text.strip():          # never cache an empty reply
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self._cache_path(messages, system).write_text(json.dumps({"text": c.text}))
                return c
            except transport.HTTPError as e:
                last = e
                if e.status not in (408, 409, 425, 429, 500, 502, 503, 504, 529) or attempt == self.spec.max_retries:
                    break
                wait = e.retry_after or delay * (1 + random.random())
                await asyncio.sleep(min(wait, 30))
                delay *= 2
            except Exception as e:  # network errors, timeouts, malformed responses
                last = e
                if attempt == self.spec.max_retries:
                    break
                await asyncio.sleep(delay * (1 + random.random()))
                delay *= 2
        self.usage.errors += 1
        raise ProviderError(f"{self.label}: {type(last).__name__}: {str(last)[:300]}")
