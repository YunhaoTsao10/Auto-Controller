from app.config import client, OPENAI_MODEL
from app.schemas import Spec, Strategy, DynamicsModel, ControllerDesign, QCReport


def control_qc_agent(state: dict) -> dict:
    raw = state.get("raw_request", "")
    spec: Spec | None = state.get("spec")
    strategy: Strategy | None = state.get("strategy")
    model: DynamicsModel | None = state.get("model")
    controller: ControllerDesign | None = state.get("controller")

    if spec is None or strategy is None or model is None or controller is None:
        report = QCReport(
            step="control",
            reviewer="control_qc_agent",
            passed=False,
            verdict="stop",
            retry_target="user",
            severity="error",
            confidence=0.98,
            summary="Control QC could not run because spec, strategy, model, or controller is missing.",
            fail_reasons=[
                "Missing required upstream artifact(s) for control review."
            ],
            evidence=[
                f"spec_present={spec is not None}",
                f"strategy_present={strategy is not None}",
                f"model_present={model is not None}",
                f"controller_present={controller is not None}",
            ],
            internal_feedback=(
                "Do not continue. The control stage must produce a structured controller "
                "after strategy and modeling are available."
            ),
            user_feedback=(
                "The workflow could not continue because the control stage did not produce "
                "a complete structured controller."
            ),
        )
        state.setdefault("qc", []).append(report)
        state["latest_qc"] = report
        state["halt_reason"] = report.user_feedback
        return state

    system = (
        "You are the control QC reviewer in a multi-agent control design workflow.\n"
        "You do NOT redesign the controller.\n"
        "You only review whether the structured controller output is consistent with the request, the selected strategy, and the structured model.\n"
        "\n"
        "Decision rules:\n"
        "- Return verdict='pass' when the controller package is technically valid, internally consistent enough for implementation, and aligned with the selected controller family.\n"
        "- Return verdict='retry' only when the controller has a repairable control-stage issue that would likely cause ambiguity, incorrect implementation, or mismatch with the model/strategy.\n"
        "- Return verdict='stop' only when the controller is fundamentally unsupported, clearly non-deliverable, or cannot be repaired within the control stage alone.\n"
        "- Prefer pass with severity='warning' over retry when the controller is technically correct but uses a more generic or more reusable presentation than the most explicit local-equilibrium form.\n"
        "- If a controller is designed for local stabilization around a nonzero equilibrium, it is acceptable for the law to be written either:\n"
        "  (a) explicitly in deviation variables, or\n"
        "  (b) in equivalent reference-tracking form with an equilibrium bias/feedforward term,\n"
        "  provided the meaning is clear and not contradictory.\n"
        "- Treat notation differences such as omega versus d(theta)/dt as acceptable if they are mathematically equivalent and the source of the signal is reasonably inferable from the model.\n"
        "- Do not require the most specific local regulation form when a more general reference-based form is intentionally used for reusability, as long as it remains compatible with the model and task.\n"
        "- Be strict only about true inconsistencies: wrong controller family, contradictory sign conventions, missing essential bias/feedforward when required, or signals that cannot be sourced from the model.\n"
        "- Return structured output only.\n"
    )

    user = (
        f"RAW REQUEST:\n{raw}\n\n"
        f"EXTRACTED SPEC:\n{spec.model_dump_json(indent=2)}\n\n"
        f"SELECTED STRATEGY:\n{strategy.model_dump_json(indent=2)}\n\n"
        f"MODEL OUTPUT:\n{model.model_dump_json(indent=2)}\n\n"
        f"CONTROLLER OUTPUT:\n{controller.model_dump_json(indent=2)}\n\n"
        "Review the control stage and return a QCReport."
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