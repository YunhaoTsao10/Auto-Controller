from langgraph.graph import StateGraph, END

from app.agents.strategy_agent import strategy_agent
from app.agents.modeling_agent import modeling_agent
from app.agents.control_agent import control_agent


def build_app():
    agents = StateGraph(dict)

    agents.add_node("strategy", strategy_agent)
    agents.add_node("model", modeling_agent)
    agents.add_node("control", control_agent)

    agents.set_entry_point("strategy")
    agents.add_edge("strategy", "model")
    agents.add_edge("model", "control")
    agents.add_edge("control", END)

    return agents.compile()