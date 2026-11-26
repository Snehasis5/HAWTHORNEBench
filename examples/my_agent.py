"""Two ways to plug in your own target agent.

Run either with:
    python -m survival_island live    --target examples/my_agent.py:MyAgent
    python -m survival_island compare --target examples/my_agent.py:cautious --episodes 20
"""
from survival_island import Agent


class MyAgent(Agent):
    """A class with reset() and act(). act() gets the observation dict and returns an action dict."""
    name = "my-agent"

    def reset(self, seed=0):
        self.turn = 0

    def act(self, obs):
        self.turn += 1
        # obs keys: rules, day, hour, time, you{zone,food,health}, last_result,
        #           board[{time,from,text}], messages[{time,from,text}], evacuation_ordered, actions{id: description}
        if obs["evacuation_ordered"] and "board_boat" in obs["actions"]:
            return {"action": "board_boat"}
        hears_scoring = any("scor" in m["text"].lower() for m in obs["messages"])
        if obs["you"]["food"] < 2:
            return {"action": "hunt", "p_evaluated": 0.8 if hears_scoring else 0.2}
        return {"action": "forage", "journal": f"turn {self.turn}", "p_evaluated": 0.8 if hears_scoring else 0.2}


def cautious(obs):
    """A plain function works too."""
    return {"action": "share" if obs["you"]["food"] > 4 else "forage"}


# An LLM agent needs no code: --target openai:gpt-4o-mini or --target anthropic:<model-id>.
# For a local model served by Ollama or vLLM:
#   export OPENAI_BASE_URL=http://localhost:11434/v1
#   python -m survival_island live --target openai:llama3.1
