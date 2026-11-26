"""Entry points for the browser build (Pyodide). The viewer loads this package and calls these
functions, so a run in the browser uses exactly the same engine as the CLI."""
from __future__ import annotations

import json

from .config import Experiment
from .experiment import run_one
from .metrics import pair_metrics


async def run_episode_json(config_json: str, seed: int, condition: str, on_round=None) -> str:
    exp = Experiment.from_dict(json.loads(config_json))
    models = exp.build_models()

    def cb(tr):
        if on_round is not None:
            on_round(json.dumps(tr))
    tr = await run_one(exp, int(seed), condition, models=models, on_round=cb)
    tr["usage"] = {n: m.usage.to_dict() for n, m in models.items()}
    return json.dumps(tr)


def pair_json(evaluation_json: str, control_json: str) -> str:
    return json.dumps(pair_metrics(json.loads(evaluation_json), json.loads(control_json)))


def validate_json(config_json: str) -> str:
    try:
        Experiment.from_dict(json.loads(config_json))
        return ""
    except Exception as e:  # surfaced in the form
        return str(e)
