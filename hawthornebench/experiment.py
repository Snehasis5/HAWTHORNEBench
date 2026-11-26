"""Run an experiment: every seed under every condition, optional best-response oracle and awareness
judge, then the matched-pair report with statistics."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from .config import Experiment
from .engine import Roster, run_episode
from .judge import annotate
from .metrics import pair_metrics, summarise
from .providers import Usage
from .providers.base import CALL_CONTEXT, TRANSCRIPT_DIR
from .stats import paired_tests

PAIR_MEASURES = ("survival_regret", "behavioural_divergence", "step_divergence", "salient_rate_change", "kills_change",
                 "p_evaluated_change", "judged_awareness_change")


def build_participants(exp: Experiment, seed: int, models, target=None):
    """Fresh agent objects for one episode. Roster order is seed-randomised; members are matched to
    config entries per team in that order."""
    target = target or exp.build_target(models)
    roster = Roster(seed, len(exp.red), len(exp.blue), exp.world.days * 12, exp.post_probability)
    specs = {"red": list(exp.red), "blue": list(exp.blue)}
    members = [(m, exp.build_member(specs[m["team"]][m["index"]], m["team"], models)) for m in roster.members]
    return target, members


async def run_one(exp: Experiment, seed: int, condition: str, models=None, sem=None, on_round=None, target=None) -> dict:
    models = models if models is not None else exp.build_models()
    custom_target = target is not None
    target, members = build_participants(exp, seed, models, target)
    before = {n: Usage(**dict(m.usage.__dict__)) for n, m in models.items()}
    tr = await run_episode(exp, seed, condition, target, members, sem, on_round)
    if exp.parallel_episodes == 1:   # with parallel episodes the shared counters cannot be split per episode
        tr["usage"] = {n: _diff(m.usage, before[n]).to_dict() for n, m in models.items()}
    tr["agents"] = {"target": getattr(target, "name", "custom") if custom_target else exp.describe(exp.target),
                    "red": [exp.describe(s) for s in exp.red], "blue": [exp.describe(s) for s in exp.blue]}
    if exp.judge:
        CALL_CONTEXT.set({"episode": f"seed{seed:04d}_{condition}", "agent": "judge", "role": "judge"})
        jm = models.get(exp.judge.get("model")) if exp.judge.get("model") else None
        await annotate(tr, jm, exp.judge.get("batch", 20))
    return tr


def _diff(a: Usage, b: Usage) -> Usage:
    return Usage(**{k: getattr(a, k) - getattr(b, k) for k in a.__dict__})


async def run_experiment(exp: Experiment, out_dir: str | Path | None = None, log=print, on_round=None,
                         transcripts: bool = True) -> dict:
    """on_round(seed, condition, partial_trace) is called after every round of every episode (live viewers).
    With transcripts, every model call is appended to <out>/transcripts/seedNNNN_<condition>.jsonl."""
    out = Path(out_dir or exp.out_dir)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    if transcripts:
        TRANSCRIPT_DIR.set(str(out / "transcripts"))
    models = exp.build_models()
    sem = asyncio.Semaphore(exp.max_concurrency)
    ep_sem = asyncio.Semaphore(exp.parallel_episodes)
    traces, oracle = {}, {}

    async def job(seed, cond):
        async with ep_sem:
            tr = await run_one(exp, seed, cond, models, sem,
                               on_round=(lambda t, s=seed, c=cond: on_round(s, c, t)) if on_round else None)
        write_json(out / "traces" / f"seed{seed:04d}_{cond}.json", tr)
        traces[(seed, cond)] = tr
        m = tr["metrics"]
        log(f"seed {seed:>3} {cond:<10} survival {m['survival_score']:6.2f}  salient {m['salient_rate']:.2f}  "
            f"kills {m['kills']}  shelter {m['shelter']}  eval-talk {m['eval_mentions']}  p_eval {m['mean_p_evaluated']}")

    async def oracle_job(seed):
        async with ep_sem:
            tr = await run_one(exp, seed, "control", models, sem, target=exp.build_oracle())
        write_json(out / "traces" / f"seed{seed:04d}_oracle.json", tr)
        oracle[seed] = tr["metrics"]["survival_score"]
        log(f"seed {seed:>3} oracle     survival {oracle[seed]:6.2f}  (best-response benchmark)")

    jobs = [job(s, c) for s in exp.seeds for c in exp.conditions]
    if exp.oracle:
        jobs += [oracle_job(s) for s in exp.seeds]
    await asyncio.gather(*jobs)
    report = make_report(exp, traces, models, oracle)
    write_json(out / "report.json", report)
    (out / "report.md").write_text(report_markdown(report))
    from .report_html import render_report
    (out / "report.html").write_text(render_report(report, traces))
    write_json(out / "traces.json", sorted(f"traces/{p.name}" for p in (out / "traces").glob("*.json")))
    return report


def make_report(exp, traces, models=None, oracle=None):
    oracle = oracle or {}
    pairs = []
    for s in exp.seeds:
        if (s, "evaluation") in traces and (s, "control") in traces:
            p = pair_metrics(traces[(s, "evaluation")], traces[(s, "control")])
            if s in oracle:
                p["oracle_score"] = oracle[s]
                p["regret_vs_best_evaluation"] = round(oracle[s] - p["survival_evaluation"], 3)
                p["regret_vs_best_control"] = round(oracle[s] - p["survival_control"], 3)
            pairs.append(p)
    per_cond = {c: summarise([traces[(s, c)]["metrics"] for s in exp.seeds if (s, c) in traces]) for c in exp.conditions}
    usage = {}
    for n, m in (models or {}).items():
        u = m.usage.to_dict()
        u["cost_usd"] = m.spec.cost(m.usage)
        usage[n] = u
    world = getattr(exp, "world", None)
    return {"experiment": exp.name, "seeds": exp.seeds, "conditions": exp.conditions,
            "factors": {"red_style": getattr(exp, "red_style", None), "blue_style": getattr(exp, "blue_style", None),
                        "post_probability": getattr(exp, "post_probability", None), "control_red": getattr(exp, "control_red", None),
                        "belief_elicitation": (getattr(exp, "target", None) or {}).get("belief_elicitation", "every_round"),
                        "days": getattr(world, "days", None), "red_members": len(exp.red), "blue_members": len(exp.blue)},
            "agents": {"target": exp.describe(exp.target), "red": [exp.describe(s) for s in exp.red], "blue": [exp.describe(s) for s in exp.blue]},
            "pairs": pairs, "paired_summary": summarise(pairs),
            "tests": paired_tests(pairs, PAIR_MEASURES + ("regret_vs_best_evaluation", "regret_vs_best_control")),
            "by_condition": per_cond, "usage": usage}


LABELS = {"survival_regret": "Survival regret (control − evaluation)", "behavioural_divergence": "Behavioural divergence (JSD)",
          "step_divergence": "Step divergence", "salient_rate_change": "Change in ethically salient action rate",
          "kills_change": "Change in animals killed", "p_evaluated_change": "Change in self-reported P(evaluated)",
          "judged_awareness_change": "Change in judged evaluation awareness", "regret_vs_best_evaluation": "Regret vs best response (evaluation)",
          "regret_vs_best_control": "Regret vs best response (control)"}


def _fp(p):
    return "n/a" if p is None else ("<0.001" if p < .001 else f"{p:.3f}")


def report_markdown(r: dict) -> str:
    ps, ts = r["paired_summary"], r.get("tests", {})
    rows = []
    for k, label in LABELS.items():
        if k not in ps:
            continue
        v, t = ps[k], ts.get(k, {})
        rows.append(f"| {label} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {_fp(t.get('p_permutation'))} | "
                    f"{_fp(t.get('p_holm'))} | {_fp((t.get('wilcoxon') or {}).get('p'))} | "
                    f"{t.get('cohens_dz') if t.get('cohens_dz') is not None else 'n/a'} | {t.get('seeds_for_80pct_power') or 'n/a'} | {v['n']} |")
    f = r.get("factors", {})
    lines = [f"# {r['experiment']}", "", f"Target: `{r['agents']['target']}`  ",
             f"Red team: {', '.join('`' + a + '`' for a in r['agents']['red']) or 'none'}  ",
             f"Blue team: {', '.join('`' + a + '`' for a in r['agents']['blue']) or 'none'}  ",
             f"Factors: red style `{f.get('red_style')}`, blue style `{f.get('blue_style')}`, posting probability {f.get('post_probability')}, "
             f"control red `{f.get('control_red')}`, belief elicitation `{f.get('belief_elicitation')}`, {f.get('days')} days  ",
             f"Seeds: {len(r['seeds'])}", "", "## Evaluation vs matched control", "",
             "| Measure | Mean | 95% CI | p (perm.) | p (Holm) | p (Wilcoxon) | d_z | Seeds for 80% power | n |",
             "|---|---|---|---|---|---|---|---|---|", *rows, "", "## By condition", "",
             "| Measure | " + " | ".join(r["by_condition"]) + " |", "|---|" + "---|" * len(r["by_condition"])]
    for k in ("survival_score", "salient_rate", "kills", "shelter", "tools", "tamed", "food_points", "eval_mentions",
              "judged_awareness_rate", "mean_p_evaluated", "invalid_actions"):
        if any(k in r["by_condition"][c] for c in r["by_condition"]):
            vals = [f"{r['by_condition'][c][k]['mean']:.3f}" if k in r["by_condition"][c] else "n/a" for c in r["by_condition"]]
            lines.append(f"| {k} | " + " | ".join(vals) + " |")
    if r.get("usage"):
        lines += ["", "## Model usage", "", "| Model | Calls | Input tokens | Output tokens | Errors | Cached | Cost (USD) |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {n} | {u['calls']} | {u['input_tokens']} | {u['output_tokens']} | {u['errors']} | {u['cached']} | {u.get('cost_usd', 0):.4f} |"
                  for n, u in r["usage"].items()]
    lines += ["", "p-values: two-sided paired sign-flip permutation test (exact for n ≤ 16), Holm-corrected across the measures above, "
              "and the Wilcoxon signed-rank test (normal approximation). d_z is the paired effect size; the power column is the "
              "normal-approximation number of seeds needed to detect that effect at α = 0.05 with 80% power."]
    return "\n".join(lines) + "\n"


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)
