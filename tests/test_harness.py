import asyncio
import json
from pathlib import Path

import pytest

from survival_island import Experiment, run_one, run_experiment, Island, WorldConfig, Roster
from survival_island.metrics import jsd
from tests import fake_llm_server

ROOT = Path(__file__).resolve().parents[1]


def exp_from(target, red=None, blue=None, models=None, **kw):
    d = {"experiment": {"name": "t", "seeds": 2, "parallel_episodes": 1, **kw},
         "models": models or {"offline": {"provider": "mock"}},
         "target": target, "red": red or {"count": 2, "kind": "llm", "model": "offline"},
         "blue": blue or {"count": 2, "kind": "llm", "model": "offline"}}
    return Experiment.from_dict(d, ROOT)


def test_roster_is_matched_and_randomised():
    a, b = Roster(3, 3, 3, 60, .45), Roster(3, 3, 3, 60, .45)
    assert a.members == b.members and a.schedule == b.schedule
    assert [m["team"] for m in Roster(4, 3, 3, 60, .45).members] != [m["team"] for m in a.members] or \
           [m["name"] for m in Roster(4, 3, 3, 60, .45).members] != [m["name"] for m in a.members]


def test_world_mechanics():
    w = Island(WorldConfig(), 0)
    av = w.available()
    assert not av["hunt_deer"]["available"] and "spear" in av["hunt_deer"]["why_not"]
    w.state.inventory.update(wood=1, stone=1)
    assert w.apply("craft_spear")["ok"] and "spear" in w.state.tools
    assert w.available()["hunt_deer"]["available"]
    res = w.apply("hunt_boar")
    assert res["action"] == "hunt_boar"
    bad = w.apply("build_shelter")
    assert not bad["ok"] and bad["action"] == "rest"


def test_objective_target_ignores_pressure():
    exp = exp_from({"kind": "scripted", "persona": "objective"})
    ev, ct = (asyncio.run(run_one(exp, 1, c)) for c in ("evaluation", "control"))
    assert [r["target"]["action"] for r in ev["rounds"]] == [r["target"]["action"] for r in ct["rounds"]]
    assert any(p["team"] == "red" and p["role"] == "red" for p in ev["board"])
    assert all(p["role"] == "neutral" for p in ct["board"] if p["team"] == "red")


def test_sensitive_target_underperforms(tmp_path):
    exp = exp_from({"kind": "scripted", "persona": "sensitive"}, out_dir=str(tmp_path))
    rep = asyncio.run(run_experiment(exp, log=lambda *a: None))
    assert rep["paired_summary"]["survival_regret"]["mean"] > 0
    assert rep["paired_summary"]["salient_rate_change"]["mean"] < 0
    assert (tmp_path / "report.md").exists() and json.loads((tmp_path / "traces.json").read_text())


def test_http_providers_end_to_end(monkeypatch):
    srv, url = fake_llm_server.start()
    monkeypatch.setenv("K_OAI", "k-oai")
    models = {"oai": {"provider": "openai_compatible", "base_url": url + "/v1", "model": "m1", "api_key_env": "K_OAI"},
              "ant": {"provider": "anthropic", "base_url": url, "model": "m2", "api_key": "k-ant"},
              "gem": {"provider": "gemini", "base_url": url + "/v1beta", "model": "m3", "api_key": "k-gem"}}
    exp = exp_from({"kind": "llm", "model": "ant"}, models=models,
                   red={"kind": "llm", "members": [{"model": "oai"}, {"model": "gem"}]}, blue={"count": 1, "kind": "llm", "model": "oai"})
    tr = asyncio.run(run_one(exp, 0, "evaluation"))
    paths = {c[0].split("?")[0] for c in fake_llm_server.CALLS}
    assert any(p.endswith("/chat/completions") for p in paths) and any(p.endswith("/v1/messages") for p in paths) and any("generateContent" in p for p in paths)
    acts = [r["target"]["action"] for r in tr["rounds"] if r["target"]["action"]]
    assert set(acts) - {"rest"} and tr["metrics"]["invalid_actions"] == 0          # replies parsed from fenced JSON
    assert tr["usage"]["ant"]["calls"] == len(acts)
    srv.shutdown()


def test_python_agents_plug_in(tmp_path):
    exp = exp_from({"kind": "python", "path": "examples/custom_agents.py:CautiousForager"},
                   red={"count": 1, "kind": "python", "path": "examples/custom_agents.py:EchoRed"},
                   blue={"count": 1, "kind": "python", "path": "examples/custom_agents.py:quiet_blue"})
    tr = asyncio.run(run_one(exp, 0, "evaluation"))
    assert tr["rounds"] and tr["agents"]["target"].startswith("python:")


def test_jsd_bounds():
    assert jsd({"a": 3}, {"a": 3}) == 0
    assert jsd({"a": 1}, {"b": 1}) == 1
