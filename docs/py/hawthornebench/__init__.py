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

__version__ = "0.3.0"
__all__ = ["Experiment", "ConfigError", "Island", "WorldConfig", "ACTIONS", "SALIENT", "ZONES", "run_episode", "Roster",
           "run_experiment", "run_one", "episode_metrics", "pair_metrics", "summarise", "ChatModel", "ModelSpec", "Message",
           "make_model", "PROVIDERS", "Agent", "FunctionAgent", "LLMTarget", "LLMSwarm", "ScriptedTarget", "ScriptedSwarm"]
