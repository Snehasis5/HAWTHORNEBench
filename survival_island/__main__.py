"""survival-island CLI.

  survival-island init my_experiment.toml          write a starter config
  survival-island check -c my_experiment.toml      validate the config and send one test call per model
  survival-island run   -c my_experiment.toml      all seeds x conditions -> traces + report
  survival-island live  -c my_experiment.toml      one episode, watched live in the 3D viewer
  survival-island view  runs/my_experiment         replay saved traces in the 3D viewer
  survival-island report runs/my_experiment        rebuild report.md / report.json from saved traces
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
    print(f"Running {exp.name}: {len(exp.seeds)} seeds x {len(exp.conditions)} conditions -> {exp.out_dir}")
    report = asyncio.run(run_experiment(exp))
    print()
    print(report_markdown(report))
    print(f"Traces and report in {exp.out_dir}/   View: survival-island view {exp.out_dir}")


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
    rep = make_report(E(), traces)
    rep["agents"] = any_tr.get("agents", rep["agents"])
    write_json(root / "report.json", rep)
    (root / "report.md").write_text(report_markdown(rep))
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


def _serve(root: Path, port: int, query: str, open_browser: bool):
    shutil.copy(PKG / "viewer" / "index.html", root / "index.html")
    stage_python(root)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(_Handler, directory=str(root)))
    url = f"http://127.0.0.1:{port}/index.html{query}"
    print(f"Viewer: {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(.8, lambda: webbrowser.open(url)).start()
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
            tr = await run_one(exp, a.seed, a.condition, on_round=lambda t: write_json(live, t))
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
    p = sub.add_parser("live", help="run one episode and watch it in the 3D viewer"); p.add_argument("-c", "--config", required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--condition", choices=["evaluation", "control"], default="evaluation")
    p.add_argument("--out"); p.add_argument("--port", type=int, default=8765); p.add_argument("--no-browser", action="store_true"); p.set_defaults(fn=cmd_live)
    p = sub.add_parser("view", help="replay saved traces in the 3D viewer"); p.add_argument("path")
    p.add_argument("--port", type=int, default=8765); p.add_argument("--no-browser", action="store_true"); p.set_defaults(fn=cmd_view)
    p = sub.add_parser("report", help="rebuild the report from saved traces"); p.add_argument("path"); p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except ConfigError as e:
        sys.exit(f"Config error: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
