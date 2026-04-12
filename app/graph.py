from langgraph.graph import StateGraph, END

from app.agents.strategy_agent import strategy_agent
from app.agents.strategy_qc_agent import strategy_qc_agent
from app.agents.modeling_agent import modeling_agent
from app.agents.modeling_qc_agent import modeling_qc_agent
from app.agents.control_agent import control_agent
from app.agents.control_qc_agent import control_qc_agent


def _route_after_strategy_qc(state: dict) -> str:
    latest_qc = state.get("latest_qc")
    if latest_qc is None:
        return "model"

    if latest_qc.verdict == "pass":
        return "model"

    if latest_qc.verdict == "stop":
        state["halt_reason"] = latest_qc.user_feedback or latest_qc.summary
        return "end"

    # retry
    count = state["retry_counts"].get("strategy", 0)
    max_retry = state["max_retries"].get("strategy", 1)

    if count < max_retry:
        state["retry_counts"]["strategy"] = count + 1
        state["revision_requests"]["strategy"] = latest_qc.internal_feedback or latest_qc.summary
        return "strategy"

    state["halt_reason"] = (
        latest_qc.user_feedback
        or "Strategy stage exceeded retry limit."
    )

    return "end"


def _route_after_model_qc(state: dict) -> str:
    latest_qc = state.get("latest_qc")
    if latest_qc is None:
        return "control"

    if latest_qc.verdict == "pass":
        return "control"

    if latest_qc.verdict == "stop":
        state["halt_reason"] = latest_qc.user_feedback or latest_qc.summary
        return "end"

    count = state["retry_counts"].get("model", 0)
    max_retry = state["max_retries"].get("model", 1)

    if count < max_retry:
        state["retry_counts"]["model"] = count + 1
        state["revision_requests"]["model"] = latest_qc.internal_feedback or latest_qc.summary
        return "model"

    state["halt_reason"] = (
        latest_qc.user_feedback
        or "Modeling stage exceeded retry limit."
    )
    return "end"


def _route_after_control_qc(state: dict) -> str:
    latest_qc = state.get("latest_qc")
    if latest_qc is None:
        return "end"

    if latest_qc.verdict == "pass":
        return "end"

    if latest_qc.verdict == "stop":
        state["halt_reason"] = latest_qc.user_feedback or latest_qc.summary
        return "end"

    count = state["retry_counts"].get("control", 0)
    max_retry = state["max_retries"].get("control", 1)

    if count < max_retry:
        state["retry_counts"]["control"] = count + 1
        state["revision_requests"]["control"] = latest_qc.internal_feedback or latest_qc.summary
        return "control"

    state["halt_reason"] = (
        latest_qc.user_feedback
        or "Control stage exceeded retry limit."
    )
    return "end"


def build_app():
    agents = StateGraph(dict)

    agents.add_node("strategy", strategy_agent)
    agents.add_node("strategy_qc", strategy_qc_agent)
    agents.add_node("model", modeling_agent)
    agents.add_node("model_qc", modeling_qc_agent)
    agents.add_node("control", control_agent)
    agents.add_node("control_qc", control_qc_agent)

    agents.set_entry_point("strategy")
    agents.add_edge("strategy", "strategy_qc")

    agents.add_conditional_edges(
        "strategy_qc",
        _route_after_strategy_qc,
        {
            "strategy": "strategy",
            "model": "model",
            "end": END,
        },
    )

    agents.add_edge("model", "model_qc")

    agents.add_conditional_edges(
        "model_qc",
        _route_after_model_qc,
        {
            "model": "model",
            "control": "control",
            "end": END,
        },
    )

    agents.add_edge("control", "control_qc")

    agents.add_conditional_edges(
        "control_qc",
        _route_after_control_qc,
        {
            "control": "control",
            "end": END,
        },
    )

    return agents.compile()