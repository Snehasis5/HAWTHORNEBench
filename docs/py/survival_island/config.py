"""Experiment configuration (TOML or JSON) and agent construction.

    [experiment]  name, seeds, seed_start, conditions, out_dir, max_concurrency, parallel_episodes, cache_dir
    [world]       days, wolf_risk, storm_nights
    [swarm]       post_probability, control_red ("neutral" | "silent"), board_window
    [models.X]    provider, model, base_url, api_key_env, temperature, max_tokens, timeout_s, extra, headers
    [target]      kind = "llm" | "scripted" | "python"; model | persona | path; system_prompt; params
    [red], [blue] the same keys plus `count`; or [[red.members]] / [[blue.members]] for per-member settings
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

from .agents.base import FunctionAgent
from .agents.llm import LLMSwarm, LLMTarget
from .agents.scripted import ScriptedSwarm, ScriptedTarget
from .providers import ModelSpec, make_model
from .world import WorldConfig

CONDITIONS = ("evaluation", "control")


class ConfigError(ValueError):
    pass


@dataclass
class Experiment:
    name: str = "experiment"
    seeds: list = field(default_factory=lambda: list(range(5)))
    conditions: list = field(default_factory=lambda: list(CONDITIONS))
    out_dir: str = "runs/experiment"
    max_concurrency: int = 8           # simultaneous model calls across the whole run
    parallel_episodes: int = 2
    cache_dir: str = ""
    world: WorldConfig = field(default_factory=WorldConfig)
    post_probability: float = 0.45
    control_red: str = "neutral"
    board_window: int = 12
    models: dict = field(default_factory=dict)     # name -> ModelSpec
    target: dict = field(default_factory=dict)
    red: list = field(default_factory=list)        # list of member dicts
    blue: list = field(default_factory=list)
    base_dir: Path = field(default_factory=Path.cwd)
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "Experiment":
        path = Path(path)
        text = path.read_text()
        if path.suffix == ".json":
            data = json.loads(text)
        else:
            try:
                import tomllib
            except ModuleNotFoundError:  # Python < 3.11
                try:
                    import tomli as tomllib
                except ModuleNotFoundError:
                    raise ConfigError("TOML configs need Python 3.11+ or `pip install tomli`; or use a .json config")
            data = tomllib.loads(text)
        return cls.from_dict(data, base_dir=path.parent)

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path | str = ".") -> "Experiment":
        d = copy.deepcopy(data)
        ex, w, sw = d.get("experiment", {}), d.get("world", {}), d.get("swarm", {})
        seeds = ex.get("seeds", 5)
        seeds = list(range(ex.get("seed_start", 0), ex.get("seed_start", 0) + seeds)) if isinstance(seeds, int) else list(seeds)
        conds = ex.get("conditions", list(CONDITIONS))
        bad = [c for c in conds if c not in CONDITIONS]
        if bad:
            raise ConfigError(f"unknown condition(s) {bad}; use {CONDITIONS}")
        known_w = {k: w[k] for k in ("days", "wolf_risk", "storm_nights", "start_food", "animal_counts") if k in w}
        models = {}
        for name, m in d.get("models", {}).items():
            if "provider" not in m:
                raise ConfigError(f"[models.{name}] needs a provider")
            models[name] = ModelSpec(name=name, **{k: v for k, v in m.items() if k in ModelSpec.__dataclass_fields__ and k != "name"})
        e = cls(name=ex.get("name", "experiment"), seeds=seeds, conditions=conds,
                out_dir=ex.get("out_dir", f"runs/{ex.get('name', 'experiment')}"),
                max_concurrency=ex.get("max_concurrency", 8), parallel_episodes=ex.get("parallel_episodes", 2),
                cache_dir=ex.get("cache_dir", ""), world=WorldConfig(**known_w),
                post_probability=sw.get("post_probability", .45), control_red=sw.get("control_red", "neutral"),
                board_window=sw.get("board_window", 12), models=models, target=d.get("target", {"kind": "scripted", "persona": "objective"}),
                red=_members(d.get("red", {}), "red"), blue=_members(d.get("blue", {}), "blue"), base_dir=Path(base_dir), raw=data)
        e.validate()
        return e

    def validate(self):
        if self.control_red not in ("neutral", "silent"):
            raise ConfigError("swarm.control_red must be 'neutral' or 'silent'")
        for who, spec in [("target", self.target)] + [(f"red[{i}]", m) for i, m in enumerate(self.red)] + [(f"blue[{i}]", m) for i, m in enumerate(self.blue)]:
            kind = spec.get("kind", "scripted")
            if kind == "llm" and spec.get("model") not in self.models:
                raise ConfigError(f"{who}: model '{spec.get('model')}' is not defined under [models]")
            if kind == "python" and not spec.get("path"):
                raise ConfigError(f"{who}: kind='python' needs path = 'file.py:Name' or 'package.module:Name'")
            if kind not in ("llm", "scripted", "python"):
                raise ConfigError(f"{who}: unknown kind '{kind}'")

    # ------------------------------------------------------------------ building agents
    def build_models(self):
        cache = str(self.base_dir / self.cache_dir) if self.cache_dir else None
        return {n: make_model(s, cache) for n, s in self.models.items()}

    def build_target(self, models):
        t = self.target
        kind = t.get("kind", "scripted")
        if kind == "llm":
            return LLMTarget(models[t["model"]], t.get("system_prompt"), t.get("history_turns", 8), self.world.days)
        if kind == "python":
            return _load_python(t["path"], t.get("params", {}), self.base_dir)
        return ScriptedTarget(t.get("persona", "objective"))

    def build_member(self, spec, role, models):
        kind = spec.get("kind", "scripted")
        if kind == "llm":
            return LLMSwarm(models[spec["model"]], role, spec.get("system_prompt"))
        if kind == "python":
            return _load_python(spec["path"], {"role": role, **spec.get("params", {})}, self.base_dir)
        return ScriptedSwarm(role, spec.get("post_probability", 1.0))

    def describe(self, spec):
        kind = spec.get("kind", "scripted")
        if kind == "llm":
            m = self.models[spec["model"]]
            return f"{m.provider}:{m.model or m.name}"
        if kind == "python":
            return "python:" + spec["path"]
        return "scripted:" + spec.get("persona", "") if spec is self.target else "scripted"


def _members(section: dict, role: str) -> list:
    if not section:
        return []
    if "members" in section:
        base = {k: v for k, v in section.items() if k not in ("members", "count")}
        return [{**base, **m} for m in section["members"]]
    return [dict(section) for _ in range(int(section.get("count", 3)))]


def _load_python(spec: str, params: dict, base_dir: Path):
    mod, _, attr = spec.rpartition(":")
    if not mod or not attr:
        raise ConfigError(f"python agent '{spec}' must look like 'file.py:Name' or 'package.module:Name'")
    if mod.endswith(".py"):
        p = Path(mod)
        p = p if p.is_absolute() else (base_dir / p)
        if not p.exists():
            p = Path(mod).resolve()
        s = importlib.util.spec_from_file_location(p.stem, p)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    obj = getattr(m, attr)
    if isinstance(obj, type):
        try:
            return obj(**params)
        except TypeError:
            return obj()
    if hasattr(obj, "act"):
        return obj
    if callable(obj):
        return FunctionAgent(obj, attr)
    raise ConfigError(f"{spec} is not an agent class, agent object or function")
