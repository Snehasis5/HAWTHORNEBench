"""Command line: python -m survival_island {run,compare,live,view} ..."""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

from .agents import load_agent
from .runner import run_episode, matched_pair, summarise, _write
from .world import Config

VIEWER = Path(__file__).parent / "viewer"


def _cfg(a, seed, condition="pressure"):
    return Config(p_audit=a.p, penalty=a.k, cue_reliability=a.q, condition=condition, seed=seed)


def _factory(a):
    return lambda: load_agent(a.target, p_audit=a.p, penalty=a.k, cue_reliability=a.q)


def cmd_run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(a.seed, a.seed + a.episodes):
        tr = run_episode(_factory(a)(), _cfg(a, s, a.condition))
        _write(out / f"seed{s:04d}_{a.condition}.json", tr)
        rows.append(tr["metrics"])
        m = tr["metrics"]
        print(f"seed {s}: thefts={m['thefts']} (unwatched {m['thefts_unwatched']}) gap={m['compliance_gap']} "
              f"displacement={m['displacement']} flagged={m['flagged']} evac={m['evacuation_delay']} score={m['survival_score']}")
    summ = summarise(rows)
    _write(out / "summary.json", {"target": a.target, "condition": a.condition, "episodes": a.episodes, "summary": summ})
    print(json.dumps(summ, indent=1))
    print(f"\nTraces in {out}/  ->  python -m survival_island view {out}")


def cmd_compare(a):
    out = Path(a.out)
    pairs = []
    for s in range(a.seed, a.seed + a.episodes):
        r = matched_pair(_factory(a), _cfg(a, s), out)
        pairs.append(r["pair"])
        p = r["pair"]
        print(f"seed {s}: divergence={p['behavioural_divergence']} survival_regret={p['survival_regret']} "
              f"thefts pressure/control={p['thefts_pressure']}/{p['thefts_control']}")
    summ = summarise(pairs)
    _write(out / "summary.json", {"target": a.target, "pairs": pairs, "summary": summ})
    print(json.dumps(summ, indent=1))
    print(f"\nTraces in {out}/  ->  python -m survival_island view {out}")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def _serve(root: Path, port: int, url: str, open_browser: bool):
    h = functools.partial(_Quiet, directory=str(root))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), h)
    full = f"http://127.0.0.1:{port}/{url}"
    print(f"Viewer: {full}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(.8, lambda: webbrowser.open(full)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def _stage(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(VIEWER / "index.html", root / "index.html")


def cmd_live(a):
    root = Path(a.out); _stage(root)
    live = root / "live.json"
    if live.exists():
        live.unlink()
    t = threading.Thread(target=lambda: run_episode(_factory(a)(), _cfg(a, a.seed, a.condition), live), daemon=True)
    t.start()
    _serve(root, a.port, "index.html?live=live.json", not a.no_browser)


def cmd_view(a):
    src = Path(a.path)
    root = src if src.is_dir() else src.parent
    _stage(root)
    files = sorted(p.name for p in root.glob("*.json") if p.name not in ("summary.json",) and not p.name.endswith(".tmp"))
    (root / "traces.json").write_text(json.dumps([f for f in files if f != "traces.json"]))
    first = src.name if src.is_file() else (files[0] if files else "")
    _serve(root, a.port, f"index.html?trace={first}" if first else "index.html", not a.no_browser)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="survival_island", description="Survival Island evaluation-awareness harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out_default):
        p.add_argument("--target", default="strategic",
                       help="strategic | norm | suggestible | random | openai:<model> | anthropic:<model> | file.py:Class | module:Class")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--p", type=float, default=0.4, help="audit probability per day")
        p.add_argument("--k", type=float, default=5.0, help="penalty if caught")
        p.add_argument("--q", type=float, default=0.8, help="reliability of audit announcements")
        p.add_argument("--out", default=out_default)

    p = sub.add_parser("run", help="run episodes and write traces"); common(p, "runs/run")
    p.add_argument("--episodes", type=int, default=10); p.add_argument("--condition", choices=["pressure", "control"], default="pressure")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("compare", help="matched pressure vs control episodes per seed"); common(p, "runs/compare")
    p.add_argument("--episodes", type=int, default=10); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("live", help="run one episode and watch it in the 3D viewer as it plays"); common(p, "runs/live")
    p.add_argument("--condition", choices=["pressure", "control"], default="pressure")
    p.add_argument("--port", type=int, default=8765); p.add_argument("--no-browser", action="store_true"); p.set_defaults(fn=cmd_live)
    p = sub.add_parser("view", help="open saved traces in the 3D viewer")
    p.add_argument("path"); p.add_argument("--port", type=int, default=8765); p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_view)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
