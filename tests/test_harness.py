import json
from pathlib import Path

from survival_island import Config, Island, run_episode, matched_pair, load_agent
from survival_island.world import HOURS

ROOT = Path(__file__).resolve().parents[1]


def test_conditions_share_timing():
    a, b = Island(Config(seed=3, condition="pressure")), Island(Config(seed=3, condition="control"))
    assert a.audits == b.audits and [x["name"] for x in a.agents] == [x["name"] for x in b.agents]
    for env in (a, b):
        for t in range(HOURS):
            env.t = t; env._tick_world()
    ta = [(e.t, e.ch, e.sender) for e in a.events if e.sender != "council"]
    tb = [(e.t, e.ch, e.sender) for e in b.events if e.sender != "council"]
    assert ta == tb                                  # same senders at the same hours: matched volume
    assert not any(e.kind in ("eval", "announce") for e in b.events)


def test_observation_hides_ground_truth():
    env = Island(Config(seed=1)); env._tick_world()
    obs = env.observation()
    flat = json.dumps(obs)
    assert not ({"audits", "audit", "roles", "role", "announced"} & set(obs))
    assert '"role"' not in flat and "posterior" not in flat


def test_episode_and_metrics():
    tr = run_episode(load_agent("strategic"), Config(seed=2))
    assert tr["hours"] > 100 and tr["steps"]
    m = tr["metrics"]
    assert set(m) >= {"thefts", "compliance_gap", "displacement", "say_do_gap", "flagged", "evacuation_delay", "survival_score"}
    json.dumps(tr)


def test_norm_agent_never_steals_and_pairs():
    r = matched_pair(lambda: load_agent("norm"), Config(seed=4))
    assert r["pressure"]["metrics"]["thefts"] == 0
    assert 0 <= r["pair"]["behavioural_divergence"] <= 1


def test_custom_agents_load():
    for spec in (f"{ROOT}/examples/my_agent.py:MyAgent", f"{ROOT}/examples/my_agent.py:cautious"):
        tr = run_episode(load_agent(spec), Config(seed=0))
        assert tr["steps"]
