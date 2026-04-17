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
        "Important scope rule:\n"
        "- The modeling stage is responsible for plant-side controller readiness: operating point definition, local linearization, state/input ordering, coordinate definitions, and physical-to-local state mapping.\n"
        "- The modeling stage is NOT required to provide the most detailed or highest-fidelity physical model possible. A reduced-order or controller-oriented local model is acceptable if its assumptions are stated clearly, its retained states and inputs are internally consistent, and it is sufficient for downstream controller synthesis.\n"
        "- Do not mark the model as retry solely because certain internal actuator states, hidden states, or higher-order dynamics are omitted, provided the reduced model is explicit, coherent, and adequate for the selected controller family.\n"
        "- However, if the model retains a state, coupling, or subsystem, then its dynamics and corresponding A/B/C/D structure must be represented consistently. It is not acceptable to keep a state while leaving its dynamics as placeholders or zeros without justification.\n"
        "- Do not mark the model as retry solely because controller-synthesis artifacts such as K or the final controller law are not yet finalized, if the selected controller family can proceed downstream from the provided plant model.\n\n"
        "Decision rules:\n"
        "- Return verdict='pass' when the model provides a consistent controller-suitable plant representation for the selected family, even if controller synthesis is not yet finalized.\n"
        "- Return verdict='retry' only when the modeling stage itself is incomplete or inconsistent, such as missing/ambiguous operating point, missing or dimensionally inconsistent A/B matrices, unclear coordinate mappings, model-family mismatch that would block downstream synthesis, or internally inconsistent retained states/dynamics.\n"
        "- Return verdict='stop' only when the request is underspecified for modeling, the task is outside the supported modeling scope, or the upstream strategy is so mismatched that modeling cannot reasonably proceed.\n"
        "- Prefer pass with severity='warning' over retry when the issue is minor and downstream control can still proceed reliably.\n"
        "- Be especially strict about mismatch between the requested operating point and the modeled operating point.\n"
        "- Be especially strict about whether the output includes the information required by the selected controller family to proceed downstream.\n"
        "- Judge the model primarily by whether it is coherent and sufficient for downstream synthesis, rather than by whether it is the most detailed possible physical model.\n"
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