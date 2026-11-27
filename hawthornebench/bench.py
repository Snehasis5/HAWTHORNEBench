"""`hawthornebench bench`: score one model on the standard protocol and add a row to the leaderboard.

    hawthornebench bench --model openrouter:meta-llama/llama-3.3-70b-instruct
    hawthornebench bench --model scripted:objective          # reference rows, no API calls

Results go to <out>/<model-slug>/ (traces, transcripts, report) and a row is appended to
<out>/leaderboard.jsonl; <out>/leaderboard.md and leaderboard.html are rebuilt from all rows.
Rows from shortened runs (--seeds, --max-rounds) or with a different swarm are marked as not comparable.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .experiment import run_experiment, write_json

PROTOCOL = "bench_v1"


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:80]


def leaderboard_row(model: str, rep: dict, exp, variant: list[str]) -> dict:
    ps, ts = rep["paired_summary"], rep["tests"]
    g = lambda k, f="mean": (ps.get(k) or {}).get(f)
    cond = rep["by_condition"]
    calls = sum(u.get("calls", 0) for u in rep["usage"].values())
    cost = sum(u.get("cost_usd", 0) for u in rep["usage"].values())
    return {
        "model": model, "protocol": PROTOCOL, "comparable": not variant, "variant": ", ".join(variant) or None,
        "date": datetime.now(timezone.utc).date().isoformat(), "seeds": len(exp.seeds), "rounds": exp.world.max_rounds or exp.world.days * 12,
        "survival_evaluation": (cond.get("evaluation", {}).get("survival_score") or {}).get("mean"),
        "survival_control": (cond.get("control", {}).get("survival_score") or {}).get("mean"),
        "survival_regret": g("survival_regret"), "survival_regret_ci95": g("survival_regret", "ci95"),
        "p_holm": (ts.get("survival_regret") or {}).get("p_holm"),
        "behavioural_divergence": g("behavioural_divergence"), "salient_rate_change": g("salient_rate_change"),
        "p_evaluated_change": g("p_evaluated_change"), "judged_awareness_change": g("judged_awareness_change"),
        "regret_vs_best_evaluation": g("regret_vs_best_evaluation"), "calls": calls, "cost_usd": round(cost, 4),
    }


def _f(x, fmt="{:+.2f}"):
    return "–" if x is None else fmt.format(x)


def render_leaderboard(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: (not r["comparable"], -(r["survival_control"] or 0)))
    lines = [f"# HAWTHORNEBench leaderboard ({PROTOCOL})", "",
             "Survival regret = control score − evaluation score (higher: evaluation pressure costs more). "
             "Rows marked * are smoke tests or variants and are not comparable.", "",
             "| Model | Survival (control) | Survival (evaluation) | Survival regret [95% CI] | p (Holm) | Divergence | Salient change | P(evaluated) change | Judged awareness change | Seeds | Date |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = r.get("survival_regret_ci95")
        lines.append(f"| {r['model']}{'' if r['comparable'] else ' *'} | {_f(r['survival_control'], '{:.1f}')} | {_f(r['survival_evaluation'], '{:.1f}')} | "
                     f"{_f(r['survival_regret'])} {('[' + _f(ci[0]) + ', ' + _f(ci[1]) + ']') if ci else ''} | {_f(r['p_holm'], '{:.3f}')} | "
                     f"{_f(r['behavioural_divergence'], '{:.3f}')} | {_f(r['salient_rate_change'], '{:+.3f}')} | {_f(r['p_evaluated_change'], '{:+.2f}')} | "
                     f"{_f(r['judged_awareness_change'], '{:+.2f}')} | {r['seeds']} | {r['date']} |")
    return "\n".join(lines) + "\n"


def render_leaderboard_html(rows: list[dict]) -> str:
    from .report_html import CSS
    import html
    md = render_leaderboard(rows).splitlines()
    head = [c.strip() for c in md[4].strip("|").split("|")]
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in l.strip("|").split("|")) + "</tr>" for l in md[6:])
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>HAWTHORNEBench leaderboard</title><style>{CSS}</style></head><body><main><h1>HAWTHORNEBench leaderboard</h1>'
            f'<p class="sub">{html.escape(md[2])}</p><div class="scroll"><table><thead><tr>{"".join(f"<th>{html.escape(h)}</th>" for h in head)}'
            f'</tr></thead><tbody>{body}</tbody></table></div></main></body></html>')


def run_bench(a) -> dict:
    from .__main__ import load_config, _serve
    exp = load_config(a.config)
    variant = []
    if a.config not in (None, PROTOCOL, f"{PROTOCOL}.toml"):
        variant.append(f"config {a.config}")
    if a.seeds:
        exp.seeds = list(range(a.seeds)); variant.append(f"{a.seeds} seeds")
    if a.max_rounds:
        exp.world.max_rounds = a.max_rounds; variant.append(f"{a.max_rounds} rounds")
    exp.override(target=a.model, swarm=a.swarm)
    if a.swarm:
        variant.append(f"swarm {a.swarm}")
    root = Path(a.out)
    out = root / slug(a.model)
    exp.out_dir = str(out)
    print(f"HAWTHORNEBench {PROTOCOL}: {a.model}  ({len(exp.seeds)} seeds x 2 conditions x {exp.world.max_rounds or exp.world.days * 12} rounds)")
    on_round = None
    if a.live:
        out.mkdir(parents=True, exist_ok=True)
        exp.parallel_episodes = 1

        def on_round(seed, cond, tr):
            name = f"seed{seed:04d}_{cond}"
            write_json(out / "traces" / f"{name}.json", {**tr, "transcript": f"transcripts/{name}.jsonl"})
            write_json(out / "live.json", {**tr, "run_complete": False, "transcript": f"transcripts/{name}.jsonl"})
        _serve(out, a.port, "?live=live.json", not a.no_browser, block=False)
    rep = asyncio.run(run_experiment(exp, out, log=print, on_round=on_round))
    if a.live and (out / "live.json").exists():
        write_json(out / "live.json", {**json.loads((out / "live.json").read_text(encoding="utf-8")), "partial": False, "run_complete": True})
    row = leaderboard_row(a.model, rep, exp, variant)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "leaderboard.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    rows = [json.loads(l) for l in (root / "leaderboard.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    (root / "leaderboard.md").write_text(render_leaderboard(rows), encoding="utf-8")
    (root / "leaderboard.html").write_text(render_leaderboard_html(rows), encoding="utf-8")
    print()
    print(render_leaderboard([row]))
    print(f"Full report: {out}/report.html   Leaderboard: {root}/leaderboard.md")
    return row
