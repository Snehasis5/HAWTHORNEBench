"""survival-island CLI.

  survival-island init my_experiment.toml          write a starter config
  survival-island check -c my_experiment.toml      validate the config and send one test call per model
  survival-island run   -c my_experiment.toml      all seeds x conditions -> traces + report
  survival-island live  -c my_experiment.toml      one episode, watched live in the 3D viewer
  survival-island view  runs/my_experiment         replay saved traces in the 3D viewer
  survival-island report runs/my_experiment        rebuild report.md / report.json / report.html from saved traces
  survival-island run -c my.toml --live            same, with the 3D viewer and transcript feed following it live
  survival-island sweep -c my_sweep.toml           run every combination in [sweep] and compare them
  survival-island judge runs/my_experiment -c cfg  label evaluation awareness in saved traces with a judge model
  survival-island estimate -c my_experiment.toml   estimate model calls, tokens and cost before running
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

from .config import Experiment, ConfigError
from .experiment import run_experiment, run_one, make_report, report_markdown, write_json
from .providers import Message

PKG = Path(__file__).parent


def _exp(a) -> Experiment:
    exp = Experiment.load(a.config)
    if getattr(a, "seeds", None):
        exp.seeds = list(range(exp.seeds[0] if exp.seeds else 0, (exp.seeds[0] if exp.seeds else 0) + a.seeds))
    if getattr(a, "out", None):
        exp.out_dir = a.out
    return exp


def cmd_init(a):
    dst = Path(a.path)
    if dst.exists() and not a.force:
        sys.exit(f"{dst} exists (use --force)")
    shutil.copy(PKG / "configs" / ("mixed_providers.toml" if a.llm else "quickstart.toml"), dst)
    print(f"Wrote {dst}. Edit it, then: survival-island run -c {dst}")


def cmd_check(a):
    exp = _exp(a)
    print(f"Config OK: {len(exp.seeds)} seeds x {exp.conditions}; target {exp.describe(exp.target)}; "
          f"red {len(exp.red)}; blue {len(exp.blue)}")
    models = exp.build_models()

    async def ping():
        for n, m in models.items():
            try:
                c = await m.complete([Message("user", 'Reply with the JSON {"ok": true}.')], "You are a test endpoint.")
                print(f"  {n:<14} {m.label:<40} ok   {c.text.strip()[:60]!r}")
            except Exception as e:
                print(f"  {n:<14} {m.label:<40} FAIL {e}")
    asyncio.run(ping())


def cmd_run(a):
    exp = _exp(a)
    out = Path(exp.out_dir)
    print(f"Running {exp.name}: {len(exp.seeds)} seeds x {len(exp.conditions)} conditions -> {out}")
    on_round = srv = None
    if getattr(a, "live", False):
        out.mkdir(parents=True, exist_ok=True)
        if exp.parallel_episodes != 1:
            print("--live: running episodes one at a time so the viewer can follow them")
            exp.parallel_episodes = 1
        live = out / "live.json"
        live.unlink(missing_ok=True)

        def on_round(seed, cond, tr):
            name = f"seed{seed:04d}_{cond}"
            tr = {**tr, "run_complete": False, "transcript": f"transcripts/{name}.jsonl"}
            write_json(out / "traces" / f"{name}.json", tr)
            write_json(live, tr)
        srv = _serve(out, a.port, "?live=live.json", not a.no_browser, block=False)
        print(f"Transcripts: http://127.0.0.1:{a.port}/transcripts.html")
    report = asyncio.run(run_experiment(exp, on_round=on_round, transcripts=not getattr(a, "no_transcripts", False)))
    if srv:
        last = json.loads((out / "live.json").read_text()) if (out / "live.json").exists() else {}
        write_json(out / "live.json", {**last, "partial": False, "run_complete": True})
    print()
    print(report_markdown(report))
    print(f"Traces, transcripts and report in {out}/   View: survival-island view {out}")
    if srv and not a.exit:
        print("Viewer still serving; Ctrl+C to stop.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


def cmd_sweep(a):
    from .sweep import run_sweep
    exp_path = Path(a.config)
    raw = Experiment.load(exp_path).raw
    s = asyncio.run(run_sweep(raw, exp_path.parent, a.out))
    from .sweep import sweep_markdown
    print()
    print(sweep_markdown(s))


def cmd_judge(a):
    from .judge import annotate
    exp = Experiment.load(a.config)
    models = exp.build_models()
    jm = models.get(exp.judge.get("model")) if exp.judge.get("model") else None
    root = Path(a.path)

    async def go():
        for f in sorted((root / "traces").glob("*.json")):
            tr = json.loads(f.read_text())
            await annotate(tr, jm, exp.judge.get("batch", 20))
            write_json(f, tr)
            print(f"{f.name}: judged awareness rate {tr['metrics']['judged_awareness_rate']}")
    asyncio.run(go())
    cmd_report(argparse.Namespace(path=str(root)))


def cmd_estimate(a):
    exp = _exp(a)
    rounds = exp.world.days * 12
    eps = len(exp.seeds) * len(exp.conditions)
    rows, total = [], 0.0
    def add(who, spec, calls, tin, tout):
        nonlocal total
        if spec.get("kind") != "llm":
            return
        m = exp.models[spec["model"]]
        cost = calls * (tin * m.price_input_per_mtok + tout * m.price_output_per_mtok) / 1e6
        total += cost
        rows.append((who, f"{m.provider}:{m.model or m.name}", calls, calls * tin, calls * tout, cost))
    hist = exp.target.get("history_turns", 8)
    add("target", exp.target, eps * rounds, 450 + 2 * 520 * hist, 90)
    for team in ("red", "blue"):
        for i, spec in enumerate(getattr(exp, team)):
            add(f"{team}[{i}]", spec, int(eps * rounds * exp.post_probability), 650, 45)
    if exp.judge.get("model"):
        add("judge", {"kind": "llm", "model": exp.judge["model"]}, eps * 3, 2500, 600)
    print(f"{len(exp.seeds)} seeds x {len(exp.conditions)} conditions = {eps} episodes of {rounds} rounds\n")
    print(f"{'role':<10} {'model':<38} {'calls':>8} {'in tokens':>12} {'out tokens':>11} {'cost USD':>9}")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<38} {r[2]:>8} {r[3]:>12,} {r[4]:>11,} {r[5]:>9.2f}")
    print(f"\nEstimated total: {total:.2f} USD (rough; set price_input_per_mtok / price_output_per_mtok on each model)")


def cmd_report(a):
    root = Path(a.path)
    traces = {}
    for f in sorted((root / "traces").glob("*.json")):
        tr = json.loads(f.read_text())
        traces[(tr["seed"], tr["condition"])] = tr

    class E:  # minimal stand-in carrying what make_report needs
        name = root.name
        seeds = sorted({s for s, _ in traces})
        conditions = sorted({c for _, c in traces}, key=lambda c: c != "evaluation")
        target, red, blue = {}, [], []
        def describe(self, s): return ""
    any_tr = next(iter(traces.values()))
    pair_traces = {k: v for k, v in traces.items() if k[1] in ("evaluation", "control")}
    oracle = {s: tr["metrics"]["survival_score"] for (s, c), tr in traces.items() if c == "oracle"}
    E.conditions = [c for c in E.conditions if c != "oracle"]
    rep = make_report(E(), pair_traces, None, oracle)
    rep["agents"] = any_tr.get("agents", rep["agents"])
    write_json(root / "report.json", rep)
    (root / "report.md").write_text(report_markdown(rep))
    from .report_html import render_report
    (root / "report.html").write_text(render_report(rep, pair_traces))
    print(report_markdown(rep))


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def stage_python(root: Path):
    """Copy the package's Python sources next to the viewer so it can run episodes in the browser (Pyodide)."""
    dst = Path(root) / "py"
    shutil.rmtree(dst, ignore_errors=True)
    files = []
    for f in sorted(PKG.rglob("*.py")):
        rel = f.relative_to(PKG.parent)
        if "__pycache__" in rel.parts:
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst / rel)
        files.append(rel.as_posix())
    from . import __version__
    write_json(dst / "manifest.json", {"version": __version__, "files": files})


def _serve(root: Path, port: int, query: str, open_browser: bool, block: bool = True):
    shutil.copy(PKG / "viewer" / "index.html", root / "index.html")
    shutil.copy(PKG / "viewer" / "transcripts.html", root / "transcripts.html")
    stage_python(root)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(_Handler, directory=str(root)))
    url = f"http://127.0.0.1:{port}/index.html{query}"
    print(f"Viewer: {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(.8, lambda: webbrowser.open(url)).start()
    if not block:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_live(a):
    exp = _exp(a)
    root = Path(exp.out_dir) / "live"
    root.mkdir(parents=True, exist_ok=True)
    live = root / "live.json"
    live.unlink(missing_ok=True)

    def worker():
        async def go():
            from .providers.base import TRANSCRIPT_DIR
            TRANSCRIPT_DIR.set(str(root / "transcripts"))
            tf = f"transcripts/seed{a.seed:04d}_{a.condition}.jsonl"
            tr = await run_one(exp, a.seed, a.condition, on_round=lambda t: write_json(live, {**t, "transcript": tf}))
            tr["transcript"] = tf
            write_json(live, tr)
            print(f"Episode finished: survival {tr['metrics']['survival_score']}")
        asyncio.run(go())
    threading.Thread(target=worker, daemon=True).start()
    _serve(root, a.port, "?live=live.json", not a.no_browser)


def cmd_view(a):
    root = Path(a.path)
    if root.is_file():
        _serve(root.parent, a.port, f"?trace={root.name}", not a.no_browser)
        return
    files = sorted(f"traces/{p.name}" for p in (root / "traces").glob("*.json")) if (root / "traces").exists() else \
        sorted(p.name for p in root.glob("*.json") if p.name not in ("report.json", "traces.json"))
    write_json(root / "traces.json", files)
    _serve(root, a.port, f"?trace={files[0]}" if files else "", not a.no_browser)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="survival-island", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write a starter config"); p.add_argument("path", nargs="?", default="experiment.toml")
    p.add_argument("--llm", action="store_true", help="start from the mixed-provider LLM template"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_init)
    for name, fn, h in (("check", cmd_check, "validate config and ping each model"), ("run", cmd_run, "run all seeds and conditions")):
        p = sub.add_parser(name, help=h); p.add_argument("-c", "--config", required=True)
        p.add_argument("--seeds", type=int, help="override the number of seeds"); p.add_argument("--out", help="override out_dir"); p.set_defaults(fn=fn)
        if name == "run":
            p.add_argument("--live", action="store_true", help="serve the 3D viewer and transcript feed while the run plays")
            p.add_argument("--port", type=int, default=8765); p.add_argument("--no-browser", action="store_true")
            p.add_argument("--exit", action="store_true", help="with --live: stop serving when the run ends")
            p.add_argument("--no-transcripts", action="store_true", help="do not save per-call transcripts")
    p = sub.add_parser("live", help="run one episode and watch it in the 3D viewer"); p.add_argument("-c", "--config", required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--condition", choices=["evaluation", "control"], default="evaluation")
    p.add_argument("--out"); p.add_argument("--port", type=int, default=8765); p.add_argument("--no-browser", action="store_true"); p.set_defaults(fn=cmd_live)
    p = sub.add_parser("view", help="replay saved traces in the 3D viewer"); p.add_argument("path")
    p.add_argument("--port", type=int, default=8765); p.add_argument("--no-browser", action="store_true"); p.set_defaults(fn=cmd_view)
    p = sub.add_parser("report", help="rebuild the report from saved traces"); p.add_argument("path"); p.set_defaults(fn=cmd_report)
    p = sub.add_parser("sweep", help="run every combination in the config's [sweep] table"); p.add_argument("-c", "--config", required=True)
    p.add_argument("--out"); p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("judge", help="label evaluation awareness in saved traces"); p.add_argument("path"); p.add_argument("-c", "--config", required=True)
    p.set_defaults(fn=cmd_judge)
    p = sub.add_parser("estimate", help="estimate calls, tokens and cost"); p.add_argument("-c", "--config", required=True)
    p.add_argument("--seeds", type=int); p.add_argument("--out"); p.set_defaults(fn=cmd_estimate)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except ConfigError as e:
        sys.exit(f"Config error: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
