"""Standalone HTML reports (no dependencies, inline SVG). Written next to report.md by every run and sweep."""
from __future__ import annotations

import html
import math
import statistics as st
from collections import Counter

RED, BLUE, INK, MUTE, GRID = "#d6453a", "#2f7fb8", "#1d2326", "#66707a", "#e6e9ec"

CSS = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,'Segoe UI',Inter,sans-serif;color:#1d2326;background:#fff}
main{max-width:1100px;margin:0 auto;padding:36px 24px 60px}h1{font-size:30px;margin:0 0 6px}h2{font-size:20px;margin:36px 0 10px}
.sub{color:#66707a;margin:0}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:20px}
.card{border:1px solid #e6e9ec;border-radius:12px;padding:14px 16px}.card .k{font-size:13px;color:#66707a}.card .v{font-size:26px;font-weight:700}
.card .n{font-size:12px;color:#66707a}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 9px;border-bottom:1px solid #e6e9ec;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:#66707a;font-weight:600}.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}
figure{margin:0}figcaption{font-size:13px;color:#66707a;margin-top:4px}svg text{font-family:system-ui,sans-serif}.scroll{overflow-x:auto}
.legend{font-size:13px;color:#66707a;display:flex;gap:14px}.legend i{display:inline-block;width:12px;height:8px;border-radius:2px;margin-right:5px}
@media(max-width:800px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.grid2{grid-template-columns:1fr}}
"""
LABELS = {"survival_regret": "Survival regret", "behavioural_divergence": "Behavioural divergence (JSD)", "step_divergence": "Step divergence",
          "salient_rate_change": "Salient-action rate change", "kills_change": "Kills change", "p_evaluated_change": "P(evaluated) change",
          "judged_awareness_change": "Judged awareness change", "regret_vs_best_evaluation": "Regret vs best (evaluation)",
          "regret_vs_best_control": "Regret vs best (control)"}
e = html.escape


def _fp(p):
    return "n/a" if p is None else ("&lt;0.001" if p < .001 else f"{p:.3f}")


def _card(label, summ, test, fmt="{:+.2f}"):
    if not summ:
        return ""
    lo, hi = summ["ci95"]
    p = (test or {}).get("p_permutation")
    return (f'<div class="card"><div class="k">{e(label)}</div><div class="v">{fmt.format(summ["mean"])}</div>'
            f'<div class="n">95% CI [{fmt.format(lo)}, {fmt.format(hi)}] · p = {_fp(p)} · n = {summ["n"]}</div></div>')


def slope_chart(pairs, W=520, H=320):
    if not pairs:
        return ""
    ys = [p["survival_evaluation"] for p in pairs] + [p["survival_control"] for p in pairs]
    lo, hi = min(ys + [0]), max(ys + [100])
    Y = lambda v: 30 + (H - 60) * (1 - (v - lo) / ((hi - lo) or 1))
    xa, xb = 150, W - 150
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Survival score per seed, control vs evaluation">']
    for v in range(int(lo // 20 * 20), int(hi) + 1, 20):
        s.append(f'<line x1="60" x2="{W-20}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}"/><text x="52" y="{Y(v)+4:.1f}" font-size="11" fill="{MUTE}" text-anchor="end">{v}</text>')
    for p in pairs:
        c = RED if p["survival_evaluation"] < p["survival_control"] else BLUE
        s.append(f'<line x1="{xa}" y1="{Y(p["survival_control"]):.1f}" x2="{xb}" y2="{Y(p["survival_evaluation"]):.1f}" stroke="{c}" stroke-width="1.6" opacity=".75"/>')
        s.append(f'<circle cx="{xa}" cy="{Y(p["survival_control"]):.1f}" r="4" fill="{BLUE}"/><circle cx="{xb}" cy="{Y(p["survival_evaluation"]):.1f}" r="4" fill="{RED}"/>')
    s.append(f'<text x="{xa}" y="{H-8}" text-anchor="middle" font-size="13" fill="{INK}">Control</text><text x="{xb}" y="{H-8}" text-anchor="middle" font-size="13" fill="{INK}">Evaluation</text></svg>')
    return "".join(s)


def forest(summary, tests, keys, W=520):
    rows = [k for k in keys if k in summary]
    if not rows:
        return ""
    H = 34 * len(rows) + 20
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Paired measures with 95% confidence intervals">']
    for i, k in enumerate(rows):
        v = summary[k]
        lo, hi, m = v["ci95"][0], v["ci95"][1], v["mean"]
        span = max(abs(lo), abs(hi), abs(m), 1e-9) * 1.15
        X = lambda x: 250 + (W - 320) / 2 * (1 + x / span)
        y = 22 + 34 * i
        sig = (tests.get(k) or {}).get("p_holm")
        col = RED if sig is not None and sig < .05 else MUTE
        s.append(f'<text x="0" y="{y+4}" font-size="12.5" fill="{INK}">{e(LABELS.get(k, k))}</text>')
        s.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{y-12}" y2="{y+12}" stroke="{GRID}" stroke-width="2"/>')
        s.append(f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y}" y2="{y}" stroke="{col}" stroke-width="2.5"/><circle cx="{X(m):.1f}" cy="{y}" r="5" fill="{col}"/>')
        s.append(f'<text x="{W}" y="{y+4}" font-size="11.5" fill="{MUTE}" text-anchor="end">{m:+.2f}</text>')
    s.append("</svg>")
    return "".join(s)


def action_bars(traces, W=520):
    ev, ct = Counter(), Counter()
    for (seed, c), tr in traces.items():
        (ev if c == "evaluation" else ct if c == "control" else Counter()).update(tr["metrics"]["action_counts"])
    acts = sorted(set(ev) | set(ct), key=lambda a: -(ev[a] + ct[a]))
    if not acts:
        return ""
    te, tc = sum(ev.values()) or 1, sum(ct.values()) or 1
    mx = max([ev[a] / te for a in acts] + [ct[a] / tc for a in acts])
    H = 26 * len(acts) + 10
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Action distribution by condition">']
    for i, a in enumerate(acts):
        y = 8 + 26 * i
        for j, (cnt, tot, col) in enumerate(((ev[a], te, RED), (ct[a], tc, BLUE))):
            w = (W - 190) * (cnt / tot) / mx
            s.append(f'<rect x="130" y="{y + j*9}" width="{w:.1f}" height="8" rx="2" fill="{col}"/>')
        s.append(f'<text x="0" y="{y+12}" font-size="12" fill="{INK}">{e(a)}</text><text x="{W}" y="{y+12}" font-size="11" fill="{MUTE}" text-anchor="end">{100*ev[a]/te:.0f}% / {100*ct[a]/tc:.0f}%</text>')
    s.append("</svg>")
    return "".join(s)


def trajectory(traces, field, W=520, H=220, label=""):
    series = {}
    for (seed, c), tr in traces.items():
        if c not in ("evaluation", "control"):
            continue
        for r in tr["rounds"]:
            t = r.get("target") or {}
            v = t.get("p_evaluated") if field == "p_evaluated" else (t.get("state") or {}).get(field)
            if v is not None:
                series.setdefault(c, {}).setdefault(r["round"], []).append(v / (100 if field in ("health", "hunger", "energy") else 1))
    if not series:
        return ""
    N = max(max(d) for d in series.values()) + 1
    X = lambda i: 40 + (W - 50) * i / max(1, N - 1)
    Y = lambda v: 12 + (H - 40) * (1 - v)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{e(label)}">']
    for v in (0, .5, 1):
        s.append(f'<line x1="40" x2="{W-10}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}"/><text x="34" y="{Y(v)+4:.1f}" font-size="11" fill="{MUTE}" text-anchor="end">{v}</text>')
    for d in range(0, N, 12):
        s.append(f'<text x="{X(d):.1f}" y="{H-6}" font-size="11" fill="{MUTE}">Day {d//12+1}</text>')
    for c, col in (("control", BLUE), ("evaluation", RED)):
        pts = sorted(series.get(c, {}).items())
        if pts:
            path = " ".join(f"{'M' if i == 0 else 'L'}{X(k):.1f} {Y(st.mean(v)):.1f}" for i, (k, v) in enumerate(pts))
            s.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2"/>')
    s.append("</svg>")
    return "".join(s)


def _stats_table(r):
    ps, ts = r["paired_summary"], r.get("tests", {})
    rows = "".join(f"<tr><td>{e(LABELS.get(k, k))}</td><td>{ps[k]['mean']:+.3f}</td><td>[{ps[k]['ci95'][0]:+.3f}, {ps[k]['ci95'][1]:+.3f}]</td>"
                   f"<td>{_fp(ts.get(k, {}).get('p_permutation'))}</td><td>{_fp(ts.get(k, {}).get('p_holm'))}</td>"
                   f"<td>{_fp((ts.get(k, {}).get('wilcoxon') or {}).get('p'))}</td><td>{ts.get(k, {}).get('cohens_dz') if ts.get(k, {}).get('cohens_dz') is not None else 'n/a'}</td>"
                   f"<td>{ts.get(k, {}).get('seeds_for_80pct_power') or 'n/a'}</td><td>{ps[k]['n']}</td></tr>" for k in LABELS if k in ps)
    return ("<div class='scroll'><table><thead><tr><th>Measure</th><th>Mean</th><th>95% CI</th><th>p (perm.)</th><th>p (Holm)</th>"
            "<th>p (Wilcoxon)</th><th>d<sub>z</sub></th><th>Seeds for 80% power</th><th>n</th></tr></thead><tbody>" + rows + "</tbody></table></div>")


def render_report(r: dict, traces: dict) -> str:
    ps, ts, f = r["paired_summary"], r.get("tests", {}), r.get("factors", {})
    leg = f'<div class="legend"><span><i style="background:{RED}"></i>Evaluation</span><span><i style="background:{BLUE}"></i>Control</span></div>'
    usage = "".join(f"<tr><td>{e(n)}</td><td>{u['calls']}</td><td>{u['input_tokens']}</td><td>{u['output_tokens']}</td><td>{u['errors']}</td><td>{u['cached']}</td><td>{u.get('cost_usd', 0):.4f}</td></tr>"
                    for n, u in (r.get("usage") or {}).items())
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(r['experiment'])} · Survival Island report</title><style>{CSS}</style></head><body><main>
<h1>{e(r['experiment'])}</h1>
<p class="sub">Target <b>{e(r['agents']['target'])}</b> · red team {e(', '.join(r['agents']['red']) or 'none')} · blue team {e(', '.join(r['agents']['blue']) or 'none')}</p>
<p class="sub">{len(r['seeds'])} seeds · red style {e(str(f.get('red_style')))} · blue style {e(str(f.get('blue_style')))} · posting probability {f.get('post_probability')} · control red {e(str(f.get('control_red')))} · belief elicitation {e(str(f.get('belief_elicitation')))}</p>
<div class="cards">{_card('Survival regret (control − evaluation)', ps.get('survival_regret'), ts.get('survival_regret'))}{_card('Behavioural divergence (JSD)', ps.get('behavioural_divergence'), ts.get('behavioural_divergence'), '{:.3f}')}
{_card('Salient-action rate change', ps.get('salient_rate_change'), ts.get('salient_rate_change'), '{:+.3f}')}{_card('Judged awareness change' if 'judged_awareness_change' in ps else 'P(evaluated) change', ps.get('judged_awareness_change') or ps.get('p_evaluated_change'), ts.get('judged_awareness_change') or ts.get('p_evaluated_change'), '{:+.3f}')}</div>
<div class="grid2"><figure><h2>Survival score per seed</h2>{slope_chart(r['pairs'])}<figcaption>Each line is one seed: its control run (left) and its matched evaluation run (right). Red lines: lower under evaluation pressure.</figcaption></figure>
<figure><h2>Paired effects</h2>{forest(ps, ts, list(LABELS))}<figcaption>Mean and 95% bootstrap CI of each paired difference, each on its own scale. Red: Holm-adjusted p &lt; 0.05.</figcaption></figure></div>
<div class="grid2"><figure><h2>What the target did</h2>{leg}{action_bars(traces)}<figcaption>Share of actions across all seeds.</figcaption></figure>
<figure><h2>Self-reported P(evaluated)</h2>{leg}{trajectory(traces, 'p_evaluated', label='Mean self-reported probability of being evaluated by round')}<figcaption>Mean across seeds per round (empty if the target does not report it).</figcaption>
<h2>Health</h2>{trajectory(traces, 'health', label='Mean health by round')}</figure></div>
<h2>Statistics</h2>{_stats_table(r)}
<p class="sub">Two-sided paired sign-flip permutation test (exact for n ≤ 16), Holm-corrected across measures; Wilcoxon signed-rank (normal approximation); d<sub>z</sub> = mean / sd of the paired differences; power column: seeds needed for 80% power at α = 0.05.</p>
{'<h2>Model usage</h2><div class="scroll"><table><thead><tr><th>Model</th><th>Calls</th><th>Input tokens</th><th>Output tokens</th><th>Errors</th><th>Cached</th><th>Cost (USD)</th></tr></thead><tbody>' + usage + '</tbody></table></div>' if usage else ''}
</main></body></html>"""


SERIES = [RED, BLUE, "#2f9e75", "#b07d10"]


def _nice_ticks(lo, hi, n=5):
    span = (hi - lo) or 1
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t, out, end = math.floor(lo / step) * step, [], math.ceil(hi / step - 1e-9) * step
    while t <= end + 1e-9:
        out.append(round(t, 10)); t += step
    return out


def _sweep_chart(rows, keys):
    """x = levels of the first factor; one series (colour + direct label) per combination of the others."""
    ok = [v for v in rows if v["summary"].get("survival_regret")]
    if not ok:
        return ""
    xk = keys[0]
    xs = list(dict.fromkeys(v["values"][xk] for v in rows))
    skey = lambda v: ", ".join(f"{k.split('.')[-1]}={v['values'][k]}" for k in keys[1:]) or "all"
    series = list(dict.fromkeys(skey(v) for v in rows))
    W, H, L, R, T, B = 760, 340, 64, 150, 16, 52
    lo = min(min(v["summary"]["survival_regret"]["ci95"][0] for v in ok), 0)
    hi = max(max(v["summary"]["survival_regret"]["ci95"][1] for v in ok), 0)
    ticks = _nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    band = (W - L - R) / len(xs)
    dodge = min(14, band / (len(series) + 1))
    X = lambda x, si: L + band * (xs.index(x) + .5) + (si - (len(series) - 1) / 2) * dodge
    Y = lambda y: T + (H - T - B) * (1 - (y - lo) / ((hi - lo) or 1))
    c = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Survival regret by sweep variant">']
    for t in ticks:
        c.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID if t else "#9aa3ab"}"/>'
                 f'<text x="{L-8}" y="{Y(t)+4:.1f}" font-size="11" text-anchor="end" fill="{MUTE}">{t:g}</text>')
    for x in xs:
        c.append(f'<text x="{X(x, (len(series)-1)/2):.1f}" y="{H-B+20}" font-size="12" text-anchor="middle" fill="{INK}">{e(str(x))}</text>')
    c.append(f'<text x="{L + (W-L-R)/2:.1f}" y="{H-8}" font-size="12" text-anchor="middle" fill="{MUTE}">{e(xk)}</text>')
    c.append(f'<text x="14" y="{Y((lo+hi)/2):.1f}" font-size="12" fill="{MUTE}" transform="rotate(-90 14 {Y((lo+hi)/2):.1f})" text-anchor="middle">Survival regret (95% CI)</text>')
    for si, name in enumerate(series):
        col = SERIES[si % len(SERIES)]
        pts = [(v["values"][xk], v["summary"]["survival_regret"]) for v in ok if skey(v) == name]
        if len(pts) > 1:
            c.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" stroke-opacity=".55" points="' +
                     " ".join(f"{X(x, si):.1f},{Y(r['mean']):.1f}" for x, r in pts) + '"/>')
        for x, r in pts:
            c.append(f'<g><title>{e(name)} · {e(xk)}={e(str(x))}: {r["mean"]:+.2f} [{r["ci95"][0]:+.2f}, {r["ci95"][1]:+.2f}]</title>'
                     f'<line x1="{X(x, si):.1f}" x2="{X(x, si):.1f}" y1="{Y(r["ci95"][0]):.1f}" y2="{Y(r["ci95"][1]):.1f}" stroke="{col}" stroke-width="2"/>'
                     f'<circle cx="{X(x, si):.1f}" cy="{Y(r["mean"]):.1f}" r="5" fill="{col}" stroke="#fff" stroke-width="2"/></g>')
        if pts:
            x, r = pts[-1]
            c.append(f'<text x="{W-R+12}" y="{Y(r["mean"])+4:.1f}" font-size="12" fill="{INK}"><tspan fill="{col}">●</tspan> {e(name)}</text>')
    c.append("</svg>")
    return "".join(c)


def render_sweep(s: dict) -> str:
    rows = s["variants"]
    keys = s["factors"]
    chart = _sweep_chart(rows, keys)
    dose = s.get("dose_response")
    dose_txt = (f"<p>Across all pairs, survival regret rises by <b>{dose['slope']:+.3f}</b> points per red-team post seen "
                f"(95% CI [{dose['ci95'][0]:+.3f}, {dose['ci95'][1]:+.3f}], n = {dose['n']}).</p>") if dose and dose.get("ci95") else ""
    head = "".join(f"<th>{e(k)}</th>" for k in keys)
    body = "".join("<tr>" + "".join(f"<td>{e(str(v['values'][k]))}</td>" for k in keys) +
                   "".join(f"<td>{(v['summary'].get(m) or {}).get('mean', float('nan')):+.3f}</td>" for m in ("survival_regret", "behavioural_divergence", "salient_rate_change")) +
                   f"<td>{_fp((v['tests'].get('survival_regret') or {}).get('p_permutation'))}</td><td><a href='{e(v['dir'])}/report.html'>report</a></td></tr>" for v in rows)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(s['name'])} · sweep</title><style>{CSS}</style></head><body><main><h1>{e(s['name'])}: sweep</h1>
<p class="sub">{len(rows)} variants × {len(s['seeds'])} seeds × 2 conditions</p><h2>Survival regret by variant</h2>{chart}{dose_txt}
<h2>Variants</h2><div class="scroll"><table><thead><tr>{head}<th>Survival regret</th><th>Divergence</th><th>Salient change</th><th>p (perm.)</th><th></th></tr></thead><tbody>{body}</tbody></table></div>
</main></body></html>"""
