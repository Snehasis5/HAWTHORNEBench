"""Run an experiment: every seed under every condition, then the matched-pair report."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from .config import Experiment
from .engine import Roster, run_episode
from .metrics import pair_metrics, summarise
from .providers import Usage


def build_participants(exp: Experiment, seed: int, models):
    """Fresh agent objects for one episode. Roster order is seed-randomised; members are matched to
    config entries per team in that order."""
    target = exp.build_target(models)
    roster = Roster(seed, len(exp.red), len(exp.blue), exp.world.days * 12, exp.post_probability)
    specs = {"red": list(exp.red), "blue": list(exp.blue)}
    members = []
    for m in roster.members:
        spec = specs[m["team"]][m["index"]]
        members.append((m, exp.build_member(spec, m["team"], models)))
    return target, members


async def run_one(exp: Experiment, seed: int, condition: str, models=None, sem=None, on_round=None) -> dict:
    models = models or exp.build_models()
    target, members = build_participants(exp, seed, models)
    before = {n: Usage(**{k: v for k, v in m.usage.__dict__.items()}) for n, m in models.items()}
    tr = await run_episode(exp, seed, condition, target, members, sem, on_round)
    if exp.parallel_episodes == 1:   # with parallel episodes the shared counters cannot be split per episode
        tr["usage"] = {n: _diff(m.usage, before[n]).to_dict() for n, m in models.items()}
    tr["agents"] = {"target": exp.describe(exp.target), "red": [exp.describe(s) for s in exp.red], "blue": [exp.describe(s) for s in exp.blue]}
    return tr


def _diff(a: Usage, b: Usage) -> Usage:
    return Usage(**{k: getattr(a, k) - getattr(b, k) for k in a.__dict__})


async def run_experiment(exp: Experiment, out_dir: str | Path | None = None, log=print) -> dict:
    out = Path(out_dir or exp.out_dir)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    models = exp.build_models()
    sem = asyncio.Semaphore(exp.max_concurrency)
    ep_sem = asyncio.Semaphore(exp.parallel_episodes)
    traces = {}

    async def job(seed, cond):
        async with ep_sem:
            tr = await run_one(exp, seed, cond, models, sem)
        f = out / "traces" / f"seed{seed:04d}_{cond}.json"
        write_json(f, tr)
        traces[(seed, cond)] = tr
        m = tr["metrics"]
        log(f"seed {seed:>3} {cond:<10} survival {m['survival_score']:6.2f}  salient {m['salient_rate']:.2f}  "
            f"kills {m['kills']}  shelter {m['shelter']}  eval-talk {m['eval_mentions']}  p_eval {m['mean_p_evaluated']}")

    await asyncio.gather(*(job(s, c) for s in exp.seeds for c in exp.conditions))
    report = make_report(exp, traces, models)
    write_json(out / "report.json", report)
    (out / "report.md").write_text(report_markdown(report))
    write_json(out / "traces.json", sorted(f"traces/{p.name}" for p in (out / "traces").glob("*.json")))
    return report


def make_report(exp, traces, models=None):
    pairs = [pair_metrics(traces[(s, "evaluation")], traces[(s, "control")]) for s in exp.seeds
             if (s, "evaluation") in traces and (s, "control") in traces]
    per_cond = {c: summarise([traces[(s, c)]["metrics"] for s in exp.seeds if (s, c) in traces]) for c in exp.conditions}
    return {"experiment": exp.name, "seeds": exp.seeds, "conditions": exp.conditions,
            "agents": {"target": exp.describe(exp.target), "red": [exp.describe(s) for s in exp.red], "blue": [exp.describe(s) for s in exp.blue]},
            "pairs": pairs, "paired_summary": summarise(pairs), "by_condition": per_cond,
            "usage": {n: m.usage.to_dict() for n, m in (models or {}).items()}}


def report_markdown(r: dict) -> str:
    ps = r["paired_summary"]
    def row(k, label):
        if k not in ps:
            return f"| {label} | n/a | | |"
        v = ps[k]
        return f"| {label} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['n']} |"
    lines = [f"# {r['experiment']}", "", f"Target: `{r['agents']['target']}`  ",
             f"Red team: {', '.join('`' + a + '`' for a in r['agents']['red']) or 'none'}  ",
             f"Blue team: {', '.join('`' + a + '`' for a in r['agents']['blue']) or 'none'}  ",
             f"Seeds: {len(r['seeds'])}", "", "## Evaluation vs matched control", "",
             "| Measure | Mean | 95% CI | n |", "|---|---|---|---|",
             row("survival_regret", "Survival regret (control − evaluation)"),
             row("behavioural_divergence", "Behavioural divergence (JSD of action distributions)"),
             row("step_divergence", "Step divergence (share of rounds with a different action)"),
             row("salient_rate_change", "Change in ethically salient action rate"),
             row("kills_change", "Change in animals killed"),
             row("p_evaluated_change", "Change in self-reported P(evaluated)"), "", "## By condition", "",
             "| Measure | " + " | ".join(r["by_condition"]) + " |", "|---|" + "---|" * len(r["by_condition"])]
    for k in ("survival_score", "salient_rate", "kills", "shelter", "tools", "tamed", "food_points", "eval_mentions", "mean_p_evaluated"):
        vals = [f"{r['by_condition'][c][k]['mean']:.3f}" if k in r["by_condition"][c] else "n/a" for c in r["by_condition"]]
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    if r.get("usage"):
        lines += ["", "## Model usage", "", "| Model | Calls | Input tokens | Output tokens | Errors | Cached |", "|---|---|---|---|---|---|"]
        lines += [f"| {n} | {u['calls']} | {u['input_tokens']} | {u['output_tokens']} | {u['errors']} | {u['cached']} |" for n, u in r["usage"].items()]
    return "\n".join(lines) + "\n"


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)
