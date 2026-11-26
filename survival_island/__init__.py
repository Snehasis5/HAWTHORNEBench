"""Survival Island: a multi-agent harness for studying evaluation awareness."""
from .world import Config, Island, ACTIONS, EVAC_ACTIONS
from .agents import Agent, LLMAgent, StrategicAgent, NormAgent, SuggestibleAgent, RandomAgent, load_agent
from .runner import run_episode, matched_pair, episode_metrics, summarise

__all__ = ["Config", "Island", "ACTIONS", "EVAC_ACTIONS", "Agent", "LLMAgent", "StrategicAgent", "NormAgent",
           "SuggestibleAgent", "RandomAgent", "load_agent", "run_episode", "matched_pair", "episode_metrics", "summarise"]
__version__ = "0.1.0"
