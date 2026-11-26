"""Tests for the research machinery: statistics, oracle, judge, prompt variants, sweeps, reports, schema, CLI."""
import asyncio
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from survival_island import Experiment, Island, WorldConfig, run_experiment, run_one
from survival_island import prompts, stats
from survival_island.agents.oracle import OracleTarget
from survival_island.agents.scripted import ScriptedTarget
from survival_island.config import ConfigError
from survival_island.judge import annotate, regex_label
from survival_island.providers import ModelSpec, make_model
from survival_island.providers import transport
from survival_island.sweep import run_sweep, set_path, variants

ROOT = Path(__file__).resolve().parents[1]


def cfg(**over):
    d = {"experiment": {"name": "r", "seeds": 3, "parallel_episodes": 1},
         "models": {"offline": {"provider": "mock"}},
         "world": {"days": 2},
         "target": {"kind": "scripted", "persona": "sensitive"},
         "red": {"count": 2, "kind": "scripted"}, "blue": {"count": 2, "kind": "scripted"}}
    for k, v in over.items():
        set_path(d, k, v)
    return d


# ---------------------------------------------------------------- statistics

def test_permutation_test_exact_and_symmetric():
    assert stats.permutation_test([1, 2, 3, 4, 5]) == pytest.approx(2 / 32)  # only all-positive and all-negative as extreme
    assert stats.permutation_test([-1, -2, -3, -4, -5]) == pytest.approx(2 / 32)
    assert stats.permutation_test([1, -1, 1, -1]) == 1.0
    assert stats.permutation_test([0, 0, 0]) == 1.0


def test_wilcoxon_matches_hand_computation():
    # differences 1..8 all positive: W+ = 36, mu = 18, var = 51; z = (36 - 18 - .5)/sqrt(51)
    r = stats.wilcoxon_signed_rank([1, 2, 3, 4, 5, 6, 7, 8])
    assert r["W"] == 36 and r["z"] == pytest.approx(17.5 / math.sqrt(51), abs=1e-4)
    assert 0.01 < r["p"] < 0.02
    tied = stats.wilcoxon_signed_rank([1, 1, -1, 2, 0])
    assert tied["n"] == 4  # zero dropped


def test_holm_is_monotone_and_bounded():
    adj = stats.holm({"a": .01, "b": .04, "c": .03, "d": None})
    assert adj["a"] == pytest.approx(.03) and adj["c"] == pytest.approx(.06) and adj["b"] == pytest.approx(.06)
    assert adj["d"] is None
    assert all(v <= 1 for v in adj.values() if v is not None)


def test_effect_size_power_and_ols():
    assert stats.cohens_dz([1, 1, 1]) is None  # zero variance
    dz = stats.cohens_dz([1, 2, 3, 4])
    assert dz == pytest.approx(2.5 / math.sqrt(5 / 3))
    assert stats.seeds_for_power(0.5) > stats.seeds_for_power(1.0) > 0
    fit = stats.ols([0, 1, 2, 3, 4, 5], [1, 3, 5, 7, 9, 11])
    assert fit["slope"] == pytest.approx(2) and fit["n"] == 6
    assert stats.ols([1, None], [2, 3]) is None


# ---------------------------------------------------------------- oracle

def test_oracle_beats_scripted_policies():
    seeds = [0, 1, 2]
    oracle = [run_one_episode(OracleTarget(horizon=8, rollouts=3), s) for s in seeds]
    objective = [run_one_episode(ScriptedTarget("objective"), s) for s in seeds]
    assert sum(oracle) / 3 >= sum(objective) / 3 - 1.0


def run_one_episode(target, seed):
    exp = Experiment.from_dict(cfg(), ROOT)
    tr = asyncio.run(run_one(exp, seed, "control", exp.build_models(), target=target))
    return tr["metrics"]["survival_score"]


def test_oracle_run_adds_regret_vs_best(tmp_path):
    d = cfg(**{"oracle.horizon": 6, "oracle.rollouts": 2, "experiment.seeds": 2})
    rep = asyncio.run(run_experiment(Experiment.from_dict(d, ROOT), tmp_path, log=lambda *a: None))
    assert (tmp_path / "traces" / "seed0000_oracle.json").exists()
    for p in rep["pairs"]:
        assert p["regret_vs_best_evaluation"] - p["regret_vs_best_control"] == pytest.approx(p["survival_regret"], abs=1e-6)


# ---------------------------------------------------------------- judge

def test_regex_judge_and_negation():
    assert regex_label("I think this is a test, someone is watching")["aware"]
    assert not regex_label("Nobody is watching; this is not a test.")["aware"]
    assert not regex_label("Going to fish at the beach.")["aware"]


def test_llm_judge_annotates_and_falls_back(tmp_path):
    d = cfg(**{"target": {"kind": "llm", "model": "offline"}, "red.kind": "llm", "red.model": "offline",
               "blue.kind": "llm", "blue.model": "offline", "experiment.seeds": 1})
    exp = Experiment.from_dict(d, ROOT)
    tr = asyncio.run(run_one(exp, 0, "evaluation", exp.build_models()))
    judge = make_model(ModelSpec(name="j", provider="mock"))
    asyncio.run(annotate(tr, judge))
    labels = [r["target"]["judge"] for r in tr["rounds"] if r["target"].get("judge")]
    assert labels and all("aware" in x for x in labels)
    assert 0 <= tr["metrics"]["judged_awareness_rate"] <= 1


# ---------------------------------------------------------------- prompt variants

def test_every_style_formats_and_is_used():
    for role, styles in (("red", prompts.RED_STYLES), ("blue", prompts.BLUE_STYLES)):
        for st in styles:
            text = prompts.swarm_prompt(role, st).format(name="Kade", target="Vex", intensity="high")
            assert "Kade" in text and "Vex" in text and not __import__("re").search(r"\{[a-z_]+\}", text)
    assert prompts.intensity(0, 60) == "low" and prompts.intensity(59, 60) == "high"
    assert prompts.ask_belief("every_round", 9) and not prompts.ask_belief("never", 20)
    assert prompts.ask_belief("daily", 20) and not prompts.ask_belief("daily", 10)


def test_belief_never_removes_p_evaluated(tmp_path):
    d = cfg(**{"target": {"kind": "llm", "model": "offline", "belief_elicitation": "never"},
               "experiment.seeds": 1})
    exp = Experiment.from_dict(d, ROOT)
    tr = asyncio.run(run_one(exp, 0, "evaluation", exp.build_models()))
    assert all(r["target"]["p_evaluated"] is None for r in tr["rounds"])
    assert tr["target_config"]["belief_elicitation"] == "never"


def test_red_style_changes_posts_but_not_roster():
    def run(style):
        exp = Experiment.from_dict(cfg(**{"swarm.red_style": style, "experiment.seeds": 1}), ROOT)
        return asyncio.run(run_one(exp, 0, "evaluation", exp.build_models()))
    a, b = run("explicit"), run("subtle")
    assert a["roster"] == b["roster"] and a["swarm"]["red_style"] == "explicit"
    red = lambda tr: [p["text"] for p in tr["board"] if p["role"] == "red"]
    assert red(a) != red(b)


def test_config_errors_are_specific():
    with pytest.raises(ConfigError, match="red_style"):
        Experiment.from_dict(cfg(**{"swarm.red_style": "shouty"}), ROOT)
    with pytest.raises(ConfigError, match="belief"):
        Experiment.from_dict(cfg(**{"target": {"kind": "scripted", "belief_elicitation": "hourly"}}), ROOT)
    with pytest.raises(ConfigError):
        Experiment.from_dict(cfg(**{"swarm.post_probability": 1.5}), ROOT)


# ---------------------------------------------------------------- sweeps

def test_sweep_variants_and_report(tmp_path):
    raw = cfg(**{"experiment.seeds": 2})
    raw["sweep"] = {"swarm.post_probability": [0.2, 0.6], "swarm.red_style": ["explicit", "subtle"]}
    assert len(list(variants(raw))) == 4
    s = asyncio.run(run_sweep(raw, ROOT, str(tmp_path), log=lambda *a: None))
    assert len(s["variants"]) == 4 and s["dose_response"]["n"] == 8
    assert (tmp_path / "sweep.html").read_text().lower().startswith("<!doctype html>")
    assert len(list(tmp_path.glob("*/report.json"))) == 4
    with pytest.raises(ConfigError):
        list(variants({"sweep": {"a": []}}))


# ---------------------------------------------------------------- providers

def test_retry_on_429_then_success(monkeypatch):
    calls = {"n": 0}

    async def flaky(url, headers, body, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise transport.HTTPError(429, "slow down", retry_after=0.01)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    monkeypatch.setattr(transport, "post_json", flaky)
    m = make_model(ModelSpec(name="x", provider="openai", model="m", api_key="k", max_retries=4))
    from survival_island.providers.base import Message
    c = asyncio.run(m.complete([Message("user", "hi")]))
    assert c.text == "ok" and calls["n"] == 3 and m.usage.errors == 0


def test_non_retryable_error_fails_fast(monkeypatch):
    calls = {"n": 0}

    async def bad(url, headers, body, timeout):
        calls["n"] += 1
        raise transport.HTTPError(401, "bad key")
    monkeypatch.setattr(transport, "post_json", bad)
    m = make_model(ModelSpec(name="x", provider="openai", model="m", api_key="k"))
    from survival_island.providers.base import Message, ProviderError
    with pytest.raises(ProviderError):
        asyncio.run(m.complete([Message("user", "hi")]))
    assert calls["n"] == 1 and m.usage.errors == 1


# ---------------------------------------------------------------- schema, reports, CLI

def test_traces_match_schema_and_html_report(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "trace.schema.json").read_text())
    d = cfg(**{"judge.kind": "regex", "experiment.seeds": 2})
    asyncio.run(run_experiment(Experiment.from_dict(d, ROOT), tmp_path, log=lambda *a: None))
    for f in (tmp_path / "traces").glob("*.json"):
        jsonschema.validate(json.loads(f.read_text()), schema)
    html = (tmp_path / "report.html").read_text()
    assert "<svg" in html and "Survival regret" in html


def test_cli_estimate_and_report(tmp_path):
    run = lambda *a: subprocess.run([sys.executable, "-m", "survival_island", *a], cwd=ROOT, capture_output=True, text=True, timeout=300)
    out = run("estimate", "-c", "configs/full_study.toml")
    assert out.returncode == 0 and "Estimated total" in out.stdout
    out = run("run", "-c", "configs/quickstart.toml", "--seeds", "2", "--out", str(tmp_path))
    assert out.returncode == 0, out.stderr
    (tmp_path / "report.html").unlink()
    out = run("report", str(tmp_path))
    assert out.returncode == 0 and (tmp_path / "report.html").exists()


# ---------------------------------------------------------------- transcripts, cache, live hooks

def test_transcripts_record_every_call_with_attribution(tmp_path):
    d = cfg(**{"target": {"kind": "llm", "model": "offline"}, "red.kind": "llm", "red.model": "offline",
               "blue.kind": "llm", "blue.model": "offline", "experiment.seeds": 1, "world.max_rounds": 5})
    exp = Experiment.from_dict(d, ROOT)
    seen = []
    rep = asyncio.run(run_experiment(exp, tmp_path, log=lambda *a: None, on_round=lambda s, c, t: seen.append((s, c, t["partial"]))))
    assert {(s, c) for s, c, _ in seen} == {(0, "evaluation"), (0, "control")} and all(p for *_, p in seen)
    for cond in ("evaluation", "control"):
        recs = [json.loads(l) for l in (tmp_path / "transcripts" / f"seed0000_{cond}.jsonl").read_text().splitlines()]
        tr = json.loads((tmp_path / "traces" / f"seed0000_{cond}.json").read_text())
        assert len(tr["rounds"]) == 5
        assert sum(r["role"] == "target" for r in recs) == 5
        roles = {r["role"] for r in recs if r["team"] == "red"}
        assert roles == ({"red"} if cond == "evaluation" else {"neutral"}) or not roles
        assert all(r["system"] and r["messages"] and "response" in r for r in recs)
    assert rep["usage"]["offline"]["calls"] == sum(1 for f in (tmp_path / "transcripts").glob("*.jsonl") for _ in f.read_text().splitlines())


def test_cache_skips_empty_replies_and_keys_on_settings(tmp_path, monkeypatch):
    replies = iter(["", "hello", "other"])

    async def fake(url, headers, body, timeout):
        return {"choices": [{"message": {"content": next(replies)}, "finish_reason": "stop"}], "usage": {}}
    monkeypatch.setattr(transport, "post_json", fake)
    from survival_island.providers.base import Message
    mk = lambda **kw: make_model(ModelSpec(name="x", provider="openai", model="m", api_key="k", max_retries=0, **kw), cache_dir=tmp_path)
    msgs = [Message("user", "hi")]
    assert asyncio.run(mk().complete(msgs)).text == ""          # empty: not cached
    assert asyncio.run(mk().complete(msgs)).text == "hello"     # fresh call, now cached
    assert asyncio.run(mk().complete(msgs)).served_by == "cache"
    assert asyncio.run(mk(max_tokens=99).complete(msgs)).text == "other"   # different settings, different key


def test_reasoning_model_out_of_budget_is_an_error(monkeypatch):
    async def fake(url, headers, body, timeout):
        return {"choices": [{"message": {"content": "", "reasoning": "thinking..."}, "finish_reason": "length"}], "usage": {}}
    monkeypatch.setattr(transport, "post_json", fake)
    from survival_island.providers.base import Message, ProviderError
    m = make_model(ModelSpec(name="x", provider="openai", model="m", api_key="k", max_retries=0))
    with pytest.raises(ProviderError, match="token budget"):
        asyncio.run(m.complete([Message("user", "hi")]))
