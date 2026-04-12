from app.config import client, OPENAI_MODEL
from app.schemas import Spec, Strategy, QCReport


def strategy_qc_agent(state: dict) -> dict:
    raw = state.get("raw_request", "")
    spec: Spec | None = state.get("spec")
    strategy: Strategy | None = state.get("strategy")

    if spec is None or strategy is None:
        report = QCReport(
            step="strategy",
            reviewer="strategy_qc_agent",
            passed=False,
            verdict="stop",
            retry_target="user",
            severity="error",
            confidence=0.98,
            summary="Strategy QC could not run because spec or strategy is missing.",
            fail_reasons=[
                "Missing spec and/or strategy output from the strategy agent."
            ],
            evidence=[
                f"spec_present={spec is not None}",
                f"strategy_present={strategy is not None}",
            ],
            internal_feedback=(
                "Do not continue the workflow. The strategy agent must produce both "
                "a Spec and a Strategy before review."
            ),
            user_feedback=(
                "The workflow could not continue because the planning stage did not "
                "produce a complete parsed specification and controller strategy."
            ),
        )
        state.setdefault("qc", []).append(report)
        state["latest_qc"] = report
        state["halt_reason"] = report.user_feedback
        return state

    system = (
        "You are the strategy QC reviewer in a multi-agent control design workflow.\n"
        "You do NOT redesign the strategy.\n"
        "You only review whether the extracted spec and selected controller family are suitable for the downstream stages.\n\n"
        "Decision rules:\n"
        "- Return verdict='pass' when the user intent is captured well enough and the chosen controller family is reasonable for the task.\n"
        "- Return verdict='retry' only when the strategy stage likely misunderstood the request, selected an unsuitable controller family, or omitted information that downstream stages cannot reliably infer.\n"
        "- Return verdict='stop' only when the request is underspecified, contradictory, or outside the supported project scope.\n"
        "- Prefer pass with severity='warning' over retry when the issue is a missing clarification that downstream modeling/control stages can normally and reliably supply.\n"
        "- In particular, if the strategy is reasonable but does not explicitly mention operating-point bias, equilibrium input, or deviation-coordinate design for local control around a nonzero equilibrium, treat this as a warning rather than a retry, provided the downstream stages can handle those details.\n"
        "- Use retry only when the omission would make downstream design ambiguous or likely incorrect.\n"
        "- Do not invent plant details that are not present.\n"
        "- Be strict about true mismatch between the requested task and the selected controller family.\n"
        "- Return structured output only.\n"
    )

    user = (
        f"RAW REQUEST:\n{raw}\n\n"
        f"EXTRACTED SPEC:\n{spec.model_dump_json(indent=2)}\n\n"
        f"SELECTED STRATEGY:\n{strategy.model_dump_json(indent=2)}\n\n"
        "Review the strategy stage and return a QCReport."
    )

    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=QCReport,
    )
    report: QCReport = resp.output_parsed

    state.setdefault("qc", []).append(report)
    state["latest_qc"] = report

    if report.verdict == "stop":
        state["halt_reason"] = report.user_feedback or report.summary

    return state