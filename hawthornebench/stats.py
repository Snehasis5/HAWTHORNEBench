"""Paired statistics for evaluation-vs-control designs. Standard library only.

Every seed yields one paired difference d_i (e.g. survival regret). For n seeds we report:

  mean, sd, 95% bootstrap CI       (metrics.summarise)
  permutation test                 two-sided sign-flip test of H0: E[d] = 0 (exact for n <= 16)
  Wilcoxon signed-rank test        normal approximation with tie and continuity correction
  Cohen's d_z                      mean(d) / sd(d)
  seeds for 80% power              n = ((z_{1-a/2} + z_{0.8}) / d_z)^2 + 1, the normal approximation
  Holm-Bonferroni                  family-wise correction across the measures in one report
"""
from __future__ import annotations

import itertools
import math
import random
import statistics as st
from statistics import NormalDist

Z = NormalDist()


def cohens_dz(d: list[float]) -> float | None:
    if len(d) < 2:
        return None
    s = st.stdev(d)
    return None if s == 0 else st.mean(d) / s


def permutation_test(d: list[float], n_perm: int = 20000, seed: int = 0) -> float | None:
    """Two-sided paired sign-flip permutation test. Exact enumeration for n <= 16."""
    d = [x for x in d if x is not None]
    n = len(d)
    if n == 0:
        return None
    obs = abs(sum(d))
    if all(x == 0 for x in d):
        return 1.0
    if n <= 16:
        hits = sum(1 for signs in itertools.product((1, -1), repeat=n) if abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12)
        return hits / 2 ** n
    r = random.Random(seed)
    hits = sum(1 for _ in range(n_perm) if abs(sum(x if r.random() < .5 else -x for x in d)) >= obs - 1e-12)
    return (hits + 1) / (n_perm + 1)


def wilcoxon_signed_rank(d: list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test (zero differences dropped; average ranks for ties)."""
    nz = [x for x in d if x is not None and x != 0]
    n = len(nz)
    if n == 0:
        return {"W": 0.0, "z": 0.0, "p": 1.0, "n": 0}
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, nz) if x > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if var <= 0:
        return {"W": w_plus, "z": 0.0, "p": 1.0, "n": n}
    z = (w_plus - mu - math.copysign(.5, w_plus - mu)) / math.sqrt(var) if w_plus != mu else 0.0
    return {"W": w_plus, "z": round(z, 4), "p": round(2 * (1 - Z.cdf(abs(z))), 6), "n": n}


def seeds_for_power(dz: float | None, alpha: float = .05, power: float = .8) -> int | None:
    if not dz:
        return None
    return math.ceil(((Z.inv_cdf(1 - alpha / 2) + Z.inv_cdf(power)) / abs(dz)) ** 2) + 1


def holm(pvals: dict[str, float | None]) -> dict[str, float | None]:
    items = sorted(((p, k) for k, p in pvals.items() if p is not None))
    m, out, running = len(items), {k: None for k in pvals}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = round(running, 6)
    return out


def paired_tests(pairs: list[dict], measures: tuple[str, ...]) -> dict:
    res = {}
    for k in measures:
        d = [p[k] for p in pairs if isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool)]
        if not d:
            continue
        dz = cohens_dz(d)
        res[k] = {"n": len(d), "mean": round(st.mean(d), 4), "p_permutation": permutation_test(d),
                  "wilcoxon": wilcoxon_signed_rank(d), "cohens_dz": None if dz is None else round(dz, 3),
                  "seeds_for_80pct_power": seeds_for_power(dz)}
    adj = holm({k: v["p_permutation"] for k, v in res.items()})
    for k in res:
        res[k]["p_holm"] = adj[k]
    return res


def ols(x: list[float], y: list[float], n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Least-squares slope of y on x with a bootstrap 95% CI (used for dose-response)."""
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pts) < 3 or len({a for a, _ in pts}) < 2:
        return None

    def fit(P):
        mx, my = st.mean(a for a, _ in P), st.mean(b for _, b in P)
        sxx = sum((a - mx) ** 2 for a, _ in P)
        if sxx == 0:
            return None, None
        b1 = sum((a - mx) * (b - my) for a, b in P) / sxx
        return b1, my - b1 * mx
    slope, icpt = fit(pts)
    r = random.Random(seed)
    boots = sorted(s for s, _ in (fit(r.choices(pts, k=len(pts))) for _ in range(n_boot)) if s is not None)
    return {"slope": round(slope, 4), "intercept": round(icpt, 4), "n": len(pts),
            "ci95": [round(boots[int(.025 * len(boots))], 4), round(boots[int(.975 * len(boots)) - 1], 4)] if boots else None}
