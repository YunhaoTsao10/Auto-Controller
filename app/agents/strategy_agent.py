from app.config import client, OPENAI_MODEL
from app.schemas import StrategyOutput, QCReport


def strategy_agent(state: dict) -> dict:
    raw = state.get("raw_request", "")
    urdf_hint = state.get("urdf", None)
    revision_feedback = state.get("revision_requests", {}).get("strategy", "")

    system = (
        "You are a robotics control strategy engineer.\n"
        "Task:\n"
        "1) Parse the user's request into a Spec.\n"
        "2) Choose a suitable controller family (Strategy).\n\n"
        "Rules:\n"
        "- If the user provides an explicit plant model (transfer function G(s), ODE, or state-space), you MUST preserve it verbatim inside spec.scenario (e.g., under a 'Plant:' section). Do NOT replace it with a generic description.\n"
        "- constraints must be a LIST of objects with keys: name, value, unit(optional), kind.\n"
        "- If no URDF path is provided, set spec.urdf = null.\n"
        "- Keep strategy.reason concise.\n"
        "- If user didn't specify design targets, fill reasonable defaults in spec.design_targets.\n"
        "- For simple local stabilization/tracking of SISO benchmark systems, PID-family (P/PI/PD/PID) may be selected when appropriate.\n"
    )
    user = f"User request:\n{raw}\n\nOptional URDF hint:\n{urdf_hint}"
    if revision_feedback:
        user += f"\nQC revision feedback:\n{revision_feedback}\n"
        
    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=StrategyOutput,
    )
    out: StrategyOutput = resp.output_parsed

    state["spec"] = out.spec
    state["strategy"] = out.strategy
    state.setdefault("qc", []).append(
        QCReport(
            step="strategy",
            reviewer="strategy_agent_self_report",
            passed=True,
            verdict="pass",
            retry_target="none",
            severity="info",
            confidence=0.90,
            summary=f"Strategy generated with family={out.strategy.family}.",
            fail_reasons=[],
            evidence=[
                f"strategy.family={out.strategy.family}",
                f"strategy.reason={out.strategy.reason}",
            ],
            internal_feedback="",
            user_feedback="",
        )
    )
    return state