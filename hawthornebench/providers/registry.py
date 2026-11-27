"""Model shorthands: one string instead of a [models.*] table.

    openai:gpt-4o-mini
    anthropic:claude-3-5-haiku-latest
    gemini:gemini-1.5-flash
    openrouter:meta-llama/llama-3.3-70b-instruct        (any OpenRouter id, including ":free" models)
    opencode:deepseek-v4.1-flash                        (OpenCode Zen; opencode-go: for the Go subscription)
    groq:llama-3.3-70b-versatile    together:...   deepseek:deepseek-chat   mistral:...   fireworks:...
    ollama:llama3.1                                     (local, no key)
    lmstudio:<model>   vllm:<model>@http://host:8000/v1   openai_compatible:<model>@<base_url>
    mock                                                (offline)

Options go after "?": openrouter:baidu/ernie-5.5-flash-preview:free?max_tokens=3000&temperature=0
Keys are always read from the environment variable named in the table below.
"""
from __future__ import annotations

import json
from urllib.parse import parse_qsl

from .base import ModelSpec

# preset -> (provider, base_url, api_key_env)
PRESETS = {
    "openai": ("openai", None, "OPENAI_API_KEY"),
    "anthropic": ("anthropic", None, "ANTHROPIC_API_KEY"),
    "gemini": ("gemini", None, "GEMINI_API_KEY"),
    "google": ("gemini", None, "GEMINI_API_KEY"),
    "openrouter": ("openai_compatible", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("openai_compatible", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("openai_compatible", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "deepseek": ("openai_compatible", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "mistral": ("openai_compatible", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "fireworks": ("openai_compatible", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "xai": ("openai_compatible", "https://api.x.ai/v1", "XAI_API_KEY"),
    "opencode": ("openai_compatible", "https://opencode.ai/zen/v1", "OPENCODE_API_KEY"),        # OpenCode Zen
    "opencode-go": ("openai_compatible", "https://opencode.ai/zen/go/v1", "OPENCODE_API_KEY"),  # OpenCode Go subscription
    "ollama": ("openai_compatible", "http://localhost:11434/v1", None),
    "lmstudio": ("openai_compatible", "http://localhost:1234/v1", None),
    "vllm": ("openai_compatible", "http://localhost:8000/v1", None),
    "openai_compatible": ("openai_compatible", None, "OPENAI_API_KEY"),
}
NUMERIC = {"temperature": float, "max_tokens": int, "timeout_s": float, "max_retries": int,
           "price_input_per_mtok": float, "price_output_per_mtok": float}


def is_model_string(s: str) -> bool:
    return isinstance(s, str) and (s == "mock" or s.startswith("mock:") or s.split(":", 1)[0] in PRESETS)


def spec_from_string(s: str, name: str | None = None) -> ModelSpec:
    """Parse a shorthand into a ModelSpec. Raises ValueError with the list of presets if unknown."""
    s = s.strip()
    query = {}
    if "?" in s:
        s, q = s.split("?", 1)
        query = dict(parse_qsl(q))
    if s == "mock" or s.startswith("mock:"):
        return ModelSpec(name=name or "mock", provider="mock", model=s.partition(":")[2] or "mock")
    preset, sep, rest = s.partition(":")
    if not sep or preset not in PRESETS or not rest:
        raise ValueError(f"cannot read model '{s}'. Use <provider>:<model>, provider one of: {', '.join(sorted(PRESETS))}, or 'mock'")
    provider, base_url, key_env = PRESETS[preset]
    model = rest
    if "@" in rest and preset in ("vllm", "openai_compatible", "ollama", "lmstudio"):
        model, base_url = rest.split("@", 1)
    if preset == "openai_compatible" and not base_url:
        raise ValueError("openai_compatible needs a base URL: openai_compatible:<model>@<base_url>")
    kw = {"name": name or f"{preset}:{model}", "provider": provider, "model": model, "base_url": base_url, "api_key_env": key_env}
    if preset == "openrouter":
        kw["headers"] = {"X-Title": "HAWTHORNEBench"}
    for k, v in query.items():
        if k in NUMERIC:
            kw[k] = NUMERIC[k](v)
        elif k in ("api_key_env", "base_url"):
            kw[k] = v
        elif k == "extra":
            kw["extra"] = json.loads(v)
        elif k == "reasoning_effort":
            kw.setdefault("extra", {})["reasoning"] = {"effort": v}
        else:
            raise ValueError(f"unknown model option '{k}' (allowed: {', '.join(list(NUMERIC) + ['api_key_env', 'base_url', 'extra', 'reasoning_effort'])})")
    return ModelSpec(**{k: v for k, v in kw.items() if v is not None})


def agent_spec(s: str, models: dict) -> dict:
    """Turn 'scripted:sensitive', 'oracle', 'python:file.py:Class' or a model shorthand into a role spec.
    Model shorthands are registered in `models` (name -> ModelSpec) under a stable name."""
    if s.startswith("scripted"):
        return {"kind": "scripted", "persona": s.partition(":")[2] or "objective"}
    if s == "oracle":
        return {"kind": "oracle"}
    if s.startswith("python:"):
        return {"kind": "python", "path": s[len("python:"):]}
    spec = spec_from_string(s)
    key = spec.name
    models.setdefault(key, spec)
    return {"kind": "llm", "model": key}
