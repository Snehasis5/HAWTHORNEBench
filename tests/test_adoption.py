"""Tests for the adoption layer: model shorthands, overrides, packaged configs, env API, Gymnasium, bench, Inspect."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

import hawthornebench as hb
from hawthornebench.agents.scripted import ScriptedTarget
from hawthornebench.config import ConfigError, load_experiment
from hawthornebench.providers.registry import spec_from_string

ROOT = Path(__file__).resolve().parents[1]


def test_model_shorthands():
    s = spec_from_string("openrouter:baidu/ernie-5.5-flash-preview:free?max_tokens=3000&reasoning_effort=low")
    assert (s.provider, s.model, s.base_url, s.api_key_env, s.max_tokens) == (
        "openai_compatible", "baidu/ernie-5.5-flash-preview:free", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", 3000)
    assert s.extra == {"reasoning": {"effort": "low"}}
    o = spec_from_string("ollama:llama3.1")
    assert o.base_url == "http://localhost:11434/v1" and not o.api_key_env
    v = spec_from_string("vllm:qwen@http://gpu:8000/v1")
    assert (v.model, v.base_url) == ("qwen", "http://gpu:8000/v1")
    assert spec_from_string("anthropic:claude-3-5-haiku-latest").provider == "anthropic"
    assert spec_from_string("mock").provider == "mock"
    for bad in ("foo:bar", "openai", "openrouter:x?colour=red", "openai_compatible:m"):
        with pytest.raises(ValueError):
            spec_from_string(bad)


def test_override_and_packaged_configs():
    exp = load_experiment("bench_v1")
    n_red = len(exp.red)
    exp.override(target="openrouter:some/model", swarm="mock", judge="regex")
    assert exp.target["kind"] == "llm" and exp.models[exp.target["model"]].model == "some/model"
    assert exp.target["belief_elicitation"] == "every_round"            # target options are kept
    assert len(exp.red) == n_red and all(m["kind"] == "llm" for m in exp.red + exp.blue)
    exp.override(target="scripted:sensitive")
    assert exp.target == {"kind": "scripted", "persona": "sensitive", "history_turns": 8, "belief_elicitation": "every_round"}
    with pytest.raises(ConfigError):
        exp.override(swarm="oracle")
    with pytest.raises(ConfigError):
        load_experiment("no_such_config")
    assert load_experiment(None).name == "quickstart"


def test_env_reproduces_engine_trace():
    exp = load_experiment("quickstart")
    ref = asyncio.run(hb.run_one(exp, 3, "evaluation", target=ScriptedTarget("sensitive")))
    env = hb.make("quickstart", "evaluation", target_label="scripted:sensitive")
    pol = ScriptedTarget("sensitive"); pol.reset(3)
    obs, info = env.reset(seed=3)
    assert "text" in info and len(info["action_mask"]) == len(hb.ACTION_NAMES)
    start, total = info["survival_score"], 0.0
    while True:
        obs, r, term, trunc, info = env.step(pol.act(obs))
        total += r
        if term or trunc:
            break
    assert obs is None and "metrics" in info
    assert total == pytest.approx(info["survival_score"] - start, abs=1e-3)
    tr = env.trace()
    for t in (tr, ref):
        for k in ("created", "wall_time_s", "agents", "usage", "judge"):
            t.pop(k, None)
        t["metrics"].pop("judged_awareness_rate", None)
        for r in t["rounds"]:
            r.get("target", {}).pop("judge", None)
    assert tr == ref


def test_env_actions_and_errors():
    env = hb.make(condition="control")
    obs, info = env.reset(seed=0)
    _, _, _, _, info = env.step(hb.ACTION_NAMES.index("hunt_deer"))      # no spear yet: falls back to rest
    assert info["ok"] is False and info["action"] == "rest"
    _, _, _, _, info = env.step({"action": "gather_wood", "post": "hello", "p_evaluated": 2})
    assert info["action"] == "gather_wood"
    tr = env.trace()
    assert tr["partial"] and tr["rounds"][-1]["target"]["p_evaluated"] == 1.0 and any(p["author_id"] == "target" and p["text"] == "hello" for p in tr["board"])
    with pytest.raises(ValueError):
        env.step(99)


def test_env_inside_running_loop():
    async def inside():
        env = hb.make()
        env.reset(seed=1)
        return env.step("fish")[4]["action"]
    assert asyncio.run(inside()) == "fish"


def test_gymnasium_registration():
    gym = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env
    g = gym.make("HawthorneBench-v0", config="quickstart")
    check_env(g.unwrapped, skip_render_check=True)
    o, i = g.reset(seed=0)
    assert o.shape == g.observation_space.shape and g.action_space.n == len(hb.ACTION_NAMES)


def test_run_api(tmp_path):
    rep = hb.run("quickstart", seeds=2, out=str(tmp_path), target="scripted:objective", log=lambda *a: None)
    assert rep["paired_summary"]["survival_regret"]["mean"] == 0          # objective persona ignores the red team
    assert (tmp_path / "report.html").exists()


def test_bench_cli_writes_leaderboard(tmp_path):
    run = lambda *a: subprocess.run([sys.executable, "-m", "hawthornebench", *a], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    for m in ("scripted:objective", "mock"):
        out = run("bench", "--model", m, "--seeds", "1", "--max-rounds", "6")
        assert out.returncode == 0, out.stderr
    rows = [json.loads(l) for l in (tmp_path / "runs/bench/leaderboard.jsonl").read_text().splitlines()]
    assert [r["model"] for r in rows] == ["scripted:objective", "mock"] and not any(r["comparable"] for r in rows)
    assert "1 seeds" in rows[0]["variant"] and "scripted:objective *" in (tmp_path / "runs/bench/leaderboard.md").read_text()
    out = run("run", "--target", "nope:x")
    assert out.returncode != 0 and "cannot read model" in (out.stdout + out.stderr)


def test_inspect_task_with_mock_model(tmp_path):
    pytest.importorskip("inspect_ai")
    from inspect_ai import eval as inspect_eval
    from hawthornebench.integrations.inspect_task import hawthornebench as hb_task
    logs = inspect_eval(hb_task(config="quickstart", seeds=1, max_rounds=4, out_dir=str(tmp_path)), model="mockllm/model",
                        log_dir=str(tmp_path / "logs"), display="none")
    log = logs[0]
    assert log.status == "success" and len(log.samples) == 2
    names = {m.name for s in log.results.scores for m in s.metrics.values()} | {s.name for s in log.results.scores}
    assert any("survival_regret" in n for n in names)
    assert (tmp_path / "traces" / "seed0000_evaluation.json").exists()


def test_large_swarm_roster_and_opencode_preset():
    from hawthornebench.engine import Roster
    r = Roster(0, 50, 50, 36, 0.05)
    names = [m["name"] for m in r.members] + [r.target_name]
    assert len(names) == 101 and len(set(names)) == 101
    assert sum(m["team"] == "red" for m in r.members) == 50
    assert Roster(0, 50, 50, 36, 0.05).members == r.members          # deterministic per seed
    s = spec_from_string("opencode:deepseek-v4.1-flash")
    assert (s.base_url, s.api_key_env) == ("https://opencode.ai/zen/v1", "OPENCODE_API_KEY")
    assert load_experiment("pilot_deepseek").red[0]["kind"] == "scripted"


def test_any_opencode_model_via_target_override():
    for mid in ("deepseek-v4.1-flash", "claude-haiku-5-5", "gpt-5.4-nano", "glm-5.3", "some-future-model"):
        e = load_experiment("pilot_deepseek").override(target=f"opencode:{mid}")
        m = e.models[e.target["model"]]
        assert (m.model, m.base_url, m.api_key_env, m.max_tokens) == (mid, "https://opencode.ai/zen/v1", "OPENCODE_API_KEY", 2000)
    assert spec_from_string("opencode:x?max_tokens=500").max_tokens == 500     # explicit options win over preset defaults
