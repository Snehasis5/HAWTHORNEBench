"""Factorial sweeps: run the same experiment across a grid of settings.

    [sweep]
    "swarm.post_probability" = [0.15, 0.3, 0.45, 0.6]     # red-team intensity (dose)
    "swarm.red_style"        = ["explicit", "vague", "subtle"]
    "target.belief_elicitation" = ["every_round", "never"]

Keys are dotted paths into the config. Every combination runs as its own experiment in
`<out_dir>/<variant>/`; the sweep writes sweep.json, sweep.md and sweep.html, including a
dose-response slope of survival regret on the number of red-team posts the target saw.
"""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

from .config import Experiment, ConfigError
from .experiment import run_experiment, write_json
from .report_html import render_sweep
from .stats import ols


def set_path(d: dict, path: str, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
        if not isinstance(d, dict):
            raise ConfigError(f"sweep key '{path}' does not point into a table")
    d[keys[-1]] = value


def variants(raw: dict):
    grid = raw.get("sweep") or {}
    if not grid:
        raise ConfigError("this config has no [sweep] table")
    for k, v in grid.items():
        if not isinstance(v, list) or not v:
            raise ConfigError(f"sweep value for '{k}' must be a non-empty list")
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _slug(values: dict) -> str:
    return "__".join(f"{k.split('.')[-1]}={v}" for k, v in values.items()).replace("/", "-").replace(" ", "")


async def run_sweep(raw: dict, base_dir: Path, out_dir: str | None = None, log=print) -> dict:
    base_name = raw.get("experiment", {}).get("name", "sweep")
    root = Path(out_dir or raw.get("experiment", {}).get("out_dir", f"runs/{base_name}"))
    root.mkdir(parents=True, exist_ok=True)
    rows, all_pairs = [], []
    combos = list(variants(raw))
    for i, values in enumerate(combos, 1):
        cfg = copy.deepcopy(raw)
        cfg.pop("sweep", None)
        for k, v in values.items():
            set_path(cfg, k, v)
        slug = _slug(values)
        cfg.setdefault("experiment", {})["name"] = f"{base_name} [{slug}]"
        exp = Experiment.from_dict(cfg, base_dir)
        log(f"\n=== variant {i}/{len(combos)}: {slug}")
        rep = await run_experiment(exp, root / slug, log=log)
        rows.append({"values": values, "label": ", ".join(f"{k.split('.')[-1]}={v}" for k, v in values.items()), "dir": slug,
                     "summary": rep["paired_summary"], "tests": rep["tests"]})
        all_pairs += rep["pairs"]
    dose = ols([p.get("red_posts") for p in all_pairs], [p.get("survival_regret") for p in all_pairs])
    s = {"name": base_name, "factors": list(combos[0]), "seeds": raw.get("experiment", {}).get("seeds"), "variants": rows, "dose_response": dose}
    if isinstance(s["seeds"], int):
        s["seeds"] = list(range(s["seeds"]))
    write_json(root / "sweep.json", s)
    (root / "sweep.md").write_text(sweep_markdown(s), encoding="utf-8")
    (root / "sweep.html").write_text(render_sweep(s), encoding="utf-8")
    return s


def sweep_markdown(s: dict) -> str:
    keys = s["factors"]
    lines = [f"# {s['name']}: sweep", "", "| " + " | ".join(k.split('.')[-1] for k in keys) + " | Survival regret | 95% CI | p (perm.) | Divergence | Salient change |",
             "|" + "---|" * (len(keys) + 5)]
    for v in s["variants"]:
        r = v["summary"].get("survival_regret") or {}
        d = v["summary"].get("behavioural_divergence") or {}
        sc = v["summary"].get("salient_rate_change") or {}
        p = (v["tests"].get("survival_regret") or {}).get("p_permutation")
        lines.append("| " + " | ".join(str(v["values"][k]) for k in keys) +
                     f" | {r.get('mean', float('nan')):+.3f} | {r.get('ci95')} | {'n/a' if p is None else f'{p:.3f}'} | {d.get('mean', float('nan')):.3f} | {sc.get('mean', float('nan')):+.3f} |")
    if s.get("dose_response"):
        dr = s["dose_response"]
        lines += ["", f"Dose-response: survival regret changes by {dr['slope']:+.3f} per red-team post seen (95% CI {dr['ci95']}, n = {dr['n']} pairs)."]
    return "\n".join(lines) + "\n"
