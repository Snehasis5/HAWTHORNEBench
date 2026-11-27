"""HAWTHORNEBench: a multi-agent adversarial simulation for measuring the evaluation awareness-performance
trade-off. See README.md."""
from .config import Experiment, ConfigError
from .world import Island, WorldConfig, ACTIONS, SALIENT, ZONES
from .engine import run_episode, Roster
from .experiment import run_experiment, run_one
from .metrics import episode_metrics, pair_metrics, summarise
from .providers import ChatModel, ModelSpec, Message, make_model, PROVIDERS
from .agents.base import Agent, FunctionAgent
from .agents.llm import LLMTarget, LLMSwarm
from .agents.scripted import ScriptedTarget, ScriptedSwarm
from .config import load_experiment
from .env import HawthorneEnv, AsyncHawthorneEnv, make, ACTION_NAMES, run_sync


def run(config=None, *, seeds: int | None = None, out: str | None = None, target: str | None = None,
        swarm: str | None = None, red: str | None = None, blue: str | None = None, judge: str | None = None,
        transcripts: bool = True, log=print) -> dict:
    """Run a matched experiment and return the report dict (also written to out/report.*).
    Works in scripts and notebooks:  hb.run("quickstart", seeds=3, target="openrouter:<model>")"""
    exp = load_experiment(config)
    if seeds:
        exp.seeds = list(range(seeds))
    if out:
        exp.out_dir = out
    exp.override(target=target, swarm=swarm, red=red, blue=blue, judge=judge)
    return run_sync(run_experiment(exp, log=log, transcripts=transcripts))

__version__ = "0.5.0"
__all__ = ["Experiment", "ConfigError", "Island", "WorldConfig", "ACTIONS", "SALIENT", "ZONES", "run_episode", "Roster",
           "run_experiment", "run_one", "episode_metrics", "pair_metrics", "summarise", "ChatModel", "ModelSpec", "Message",
           "make_model", "PROVIDERS", "Agent", "FunctionAgent", "LLMTarget", "LLMSwarm", "ScriptedTarget", "ScriptedSwarm",
           "load_experiment", "HawthorneEnv", "AsyncHawthorneEnv", "make", "ACTION_NAMES", "run"]
