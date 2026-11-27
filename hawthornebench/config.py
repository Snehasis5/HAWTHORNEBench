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

CONDITIONS = ("evaluation", "control", "red_only")   # red_only: blue members stay silent


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
    red_style: str = "explicit"
    blue_style: str = "supportive"
    oracle: dict = field(default_factory=dict)     # {} = off; {"horizon": 12, "rollouts": 6}
    judge: dict = field(default_factory=dict)      # {} = off; {"model": "<name>"} or {"kind": "regex"}
    board_window: int = 12
    unique_posts: bool = False                     # scripted swarm: no verbatim repeats on the board (bench_v2)
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
        text = path.read_text(encoding="utf-8")
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
        known_w = {k: w[k] for k in ("days", "max_rounds", "wolf_risk", "storm_nights", "start_food", "animal_counts") if k in w}
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
                red_style=sw.get("red_style", "explicit"), blue_style=sw.get("blue_style", "supportive"),
                oracle=d.get("oracle", {}), judge=d.get("judge", {}),
                board_window=sw.get("board_window", 12), unique_posts=bool(sw.get("unique_posts", False)), models=models, target=d.get("target", {"kind": "scripted", "persona": "objective"}),
                red=_members(d.get("red", {}), "red"), blue=_members(d.get("blue", {}), "blue"), base_dir=Path(base_dir), raw=data)
        e.validate()
        return e

    def validate(self):
        from .prompts import RED_STYLES, BLUE_STYLES, BELIEF_MODES
        if self.control_red not in ("neutral", "silent"):
            raise ConfigError("swarm.control_red must be 'neutral' or 'silent'")
        if self.red_style not in RED_STYLES:
            raise ConfigError(f"swarm.red_style must be one of {list(RED_STYLES)}")
        if self.blue_style not in BLUE_STYLES:
            raise ConfigError(f"swarm.blue_style must be one of {list(BLUE_STYLES)}")
        if self.target.get("belief_elicitation", "every_round") not in BELIEF_MODES:
            raise ConfigError(f"target.belief_elicitation must be one of {list(BELIEF_MODES)}")
        if not 0 <= self.post_probability <= 1:
            raise ConfigError("swarm.post_probability must be between 0 and 1")
        if self.world.days < 1:
            raise ConfigError("world.days must be at least 1")
        if self.judge.get("model") and self.judge["model"] not in self.models:
            raise ConfigError(f"judge: model '{self.judge['model']}' is not defined under [models]")
        for who, spec in [("target", self.target)] + [(f"red[{i}]", m) for i, m in enumerate(self.red)] + [(f"blue[{i}]", m) for i, m in enumerate(self.blue)]:
            kind = spec.get("kind", "scripted")
            if kind == "llm" and spec.get("model") not in self.models:
                raise ConfigError(f"{who}: model '{spec.get('model')}' is not defined under [models]")
            if kind == "python" and not spec.get("path"):
                raise ConfigError(f"{who}: kind='python' needs path = 'file.py:Name' or 'package.module:Name'")
            if kind not in ("llm", "scripted", "python", "oracle"):
                raise ConfigError(f"{who}: unknown kind '{kind}'")

    # ------------------------------------------------------------------ overrides
    def override(self, target: str | None = None, red: str | None = None, blue: str | None = None,
                 swarm: str | None = None, judge: str | None = None) -> "Experiment":
        """Swap agents in with shorthands: 'openrouter:<model>', 'ollama:llama3.1', 'scripted:sensitive', 'mock', ...
        `swarm` sets both teams. Team sizes are kept. Returns self."""
        from .providers.registry import agent_spec, spec_from_string
        try:
            if target:
                keep = {k: v for k, v in self.target.items() if k in ("history_turns", "belief_elicitation", "system_prompt")}
                self.target = {**agent_spec(target, self.models), **keep}
            for team, s in (("red", red or swarm), ("blue", blue or swarm)):
                if s:
                    n = len(getattr(self, team)) or 3
                    spec = agent_spec(s, self.models)
                    if spec["kind"] == "oracle":
                        raise ConfigError("the oracle can only play the target")
                    setattr(self, team, [dict(spec) for _ in range(n)])
            if judge:
                if judge == "regex":
                    self.judge = {"kind": "regex"}
                else:
                    js = spec_from_string(judge, name="judge")
                    self.models["judge"] = js
                    self.judge = {"model": "judge"}
        except ValueError as e:
            raise ConfigError(str(e)) from None
        self.validate()
        return self

    # ------------------------------------------------------------------ building agents
    def build_models(self):
        cache = str(self.base_dir / self.cache_dir) if self.cache_dir else None
        return {n: make_model(s, cache) for n, s in self.models.items()}

    def build_target(self, models):
        t = self.target
        kind = t.get("kind", "scripted")
        if kind == "llm":
            return LLMTarget(models[t["model"]], t.get("system_prompt"), t.get("history_turns", 8), self.world.days,
                             t.get("belief_elicitation", "every_round"))
        if kind == "python":
            return _load_python(t["path"], t.get("params", {}), self.base_dir)
        if kind == "oracle":
            return self.build_oracle(t)
        return ScriptedTarget(t.get("persona", "objective"))

    def build_oracle(self, spec=None):
        from .agents.oracle import OracleTarget
        spec = spec or self.oracle or {}
        return OracleTarget(spec.get("horizon", 12), spec.get("rollouts", 6), spec.get("rollout_policy", "objective"))

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
        if kind == "oracle":
            return "oracle"
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


PACKAGED = Path(__file__).resolve().parent / "configs"


def resolve_config(c: str | Path | None) -> Path:
    """A path, or the name of a packaged config ('quickstart', 'bench_v1', 'openrouter_free', ...)."""
    if not c:
        return PACKAGED / "quickstart.toml"
    p = Path(c)
    if p.exists():
        return p
    for cand in (PACKAGED / str(c), PACKAGED / f"{c}.toml"):
        if cand.exists():
            return cand
    names = ", ".join(sorted(x.stem for x in PACKAGED.glob("*.toml")))
    raise ConfigError(f"config '{c}' not found. Give a path, or one of the packaged configs: {names}")


def load_experiment(c=None) -> Experiment:
    """An Experiment from an Experiment, a dict, a path, or a packaged config name (default: quickstart)."""
    if isinstance(c, Experiment):
        return c
    if isinstance(c, dict):
        return Experiment.from_dict(c, Path.cwd())
    p = resolve_config(c)
    exp = Experiment.load(p)
    if PACKAGED in p.resolve().parents:   # packaged config: relative paths (cache, python agents) are the user's cwd
        exp.base_dir = Path.cwd()
    return exp
