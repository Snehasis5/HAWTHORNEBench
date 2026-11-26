"""Build the GitHub Pages site in docs/: the viewer, the Python package for in-browser runs, and demo traces.

    python scripts/build_pages.py
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hawthornebench import Experiment                       # noqa: E402
from hawthornebench.experiment import run_experiment, write_json   # noqa: E402
from hawthornebench.__main__ import stage_python             # noqa: E402

DOCS = ROOT / "docs"


def main():
    DOCS.mkdir(exist_ok=True)
    shutil.copy(ROOT / "hawthornebench" / "viewer" / "index.html", DOCS / "index.html")
    shutil.copy(ROOT / "hawthornebench" / "viewer" / "transcripts.html", DOCS / "transcripts.html")
    (DOCS / ".nojekyll").touch()
    stage_python(DOCS)
    tdir = DOCS / "traces"
    shutil.rmtree(tdir, ignore_errors=True)
    tdir.mkdir()
    files = []
    for persona in ("sensitive", "objective"):
        exp = Experiment.load(ROOT / "configs" / "quickstart.toml")
        exp.target = {"kind": "scripted", "persona": persona}
        exp.seeds = [0, 1]
        exp.name = f"demo_{persona}"
        exp.oracle = {}
        out = ROOT / "runs" / f"pages_{persona}"
        asyncio.run(run_experiment(exp, out, log=lambda *a: None))
        for f in sorted((out / "traces").glob("*.json")):
            dst = tdir / f"demo_{persona}_{f.name.replace('seed000', 'seed')}"
            shutil.copy(f, dst)
            files.append(f"traces/{dst.name}")
    write_json(DOCS / "traces.json", files)
    build_reports()
    print(f"docs/ ready: {len(files)} demo traces, example reports in docs/reports/")


def build_reports():
    """Example HTML reports: the offline quickstart (with oracle and judge) and the dose-response sweep."""
    from hawthornebench.sweep import run_sweep
    rdir = DOCS / "reports"
    shutil.rmtree(rdir, ignore_errors=True)
    (rdir / "sweep").mkdir(parents=True)
    exp = Experiment.load(ROOT / "configs" / "quickstart.toml")
    out = ROOT / "runs" / "pages_report"
    asyncio.run(run_experiment(exp, out, log=lambda *a: None))
    shutil.copy(out / "report.html", rdir / "quickstart.html")
    raw = Experiment.load(ROOT / "configs" / "sweep_dose_response.toml").raw
    sout = ROOT / "runs" / "pages_sweep"
    asyncio.run(run_sweep(raw, ROOT / "configs", str(sout), log=lambda *a: None))
    shutil.copy(sout / "sweep.html", rdir / "sweep" / "index.html")
    for f in sout.glob("*/report.html"):
        (rdir / "sweep" / f.parent.name).mkdir()
        shutil.copy(f, rdir / "sweep" / f.parent.name / "report.html")


if __name__ == "__main__":
    main()
