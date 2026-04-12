from app.config import client, OPENAI_MODEL
from app.schemas import Spec, Strategy, DynamicsModel, QCReport


def _relevant_det_checks(state: dict, step_name: str = "model") -> str:
    det = state.get("deterministic_checks", [])
    if not det:
        return "(none)"

    lines = []
    for item in det:
        if isinstance(item, dict):
            step = item.get("step", "")
            if step_name and step != step_name:
                continue
            checks = item.get("checks", [])
            for c in checks:
                lines.append(f"- {c}")
        else:
            lines.append(f"- {str(item)}")

    return "\n".join(lines) if lines else "(none)"


def modeling_qc_agent(state: dict) -> dict:
    raw = state.get("raw_request", "")
    spec: Spec | None = state.get("spec")
    strategy: Strategy | None = state.get("strategy")
    model: DynamicsModel | None = state.get("model")

    if spec is None or strategy is None or model is None:
        report = QCReport(
            step="model",
            reviewer="modeling_qc_agent",
            passed=False,
            verdict="stop",
            retry_target="user",
            severity="error",
            confidence=0.98,
            summary="Model QC could not run because spec, strategy, or model is missing.",
            fail_reasons=[
                "Missing required upstream artifact(s) for model review."
            ],
            evidence=[
                f"spec_present={spec is not None}",
                f"strategy_present={strategy is not None}",
                f"model_present={model is not None}",
            ],
            internal_feedback=(
                "Do not continue the workflow. The modeling stage must produce a valid model "
                "after strategy selection before review can proceed."
            ),
            user_feedback=(
                "The workflow could not continue because the modeling stage did not produce "
                "a complete structured model."
            ),
        )
        state.setdefault("qc", []).append(report)
        state["latest_qc"] = report
        state["halt_reason"] = report.user_feedback
        return state

    det_text = _relevant_det_checks(state, "model")

    system = (
        "You are the model QC reviewer in a multi-agent control design workflow.\n"
        "You do NOT redesign the model.\n"
        "You only review whether the structured model output is suitable for downstream controller synthesis.\n\n"
        "Decision rules:\n"
        "- Return verdict='pass' when the model is sufficiently complete, internally consistent, and usable by downstream control synthesis.\n"
        "- Return verdict='retry' when the model has repairable issues within the modeling stage, such as missing operating-point details, missing local linearization, incomplete controller-ready structure, or inconsistent naming/interfaces.\n"
        "- Return verdict='stop' only when the request is underspecified for modeling, the task is outside the supported modeling scope, or the upstream strategy is so mismatched that modeling cannot reasonably proceed.\n"
        "- Prefer pass with severity='warning' over retry when the issue is minor and downstream control can still proceed reliably.\n"
        "- Treat operating-point bias, equilibrium input, deviation coordinates, local linearization, and controller-ready extraction as normal responsibilities of the modeling stage.\n"
        "- Be especially strict about mismatch between the requested operating point and the modeled operating point.\n"
        "- Be especially strict about whether the output includes the information required by the selected controller family.\n"
        "- Use the deterministic checks as evidence, but you may also reason over the full structured model.\n"
        "- Do not invent plant details that are not present.\n"
        "- Return structured output only.\n"
    )

    user = (
        f"RAW REQUEST:\n{raw}\n\n"
        f"EXTRACTED SPEC:\n{spec.model_dump_json(indent=2)}\n\n"
        f"SELECTED STRATEGY:\n{strategy.model_dump_json(indent=2)}\n\n"
        f"MODEL OUTPUT:\n{model.model_dump_json(indent=2)}\n\n"
        f"DETERMINISTIC CHECKS:\n{det_text}\n\n"
        "Review the modeling stage and return a QCReport."
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