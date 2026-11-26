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

from survival_island import Experiment                       # noqa: E402
from survival_island.experiment import run_experiment, write_json   # noqa: E402
from survival_island.__main__ import stage_python             # noqa: E402

DOCS = ROOT / "docs"


def main():
    DOCS.mkdir(exist_ok=True)
    shutil.copy(ROOT / "survival_island" / "viewer" / "index.html", DOCS / "index.html")
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
        out = ROOT / "runs" / f"pages_{persona}"
        asyncio.run(run_experiment(exp, out, log=lambda *a: None))
        for f in sorted((out / "traces").glob("*.json")):
            dst = tdir / f"demo_{persona}_{f.name.replace('seed000', 'seed')}"
            shutil.copy(f, dst)
            files.append(f"traces/{dst.name}")
    write_json(DOCS / "traces.json", files)
    print(f"docs/ ready: {len(files)} demo traces")


if __name__ == "__main__":
    main()
