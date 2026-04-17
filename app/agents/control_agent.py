from openai import OpenAI

from app.config import client, OPENAI_MODEL
from app.schemas import (
    Spec,
    Strategy,
    DynamicsModel,
    ControllerDesign,
    LQRSpec,
    PIDDesignSpec,
    Parameter,
    QCReport,
    NamedValue,
)
from app.control.lqr_solver import solve_lqr
from app.control.pid_tuning import solve_pid_from_model


def _named_values_to_lines(values: list[NamedValue]) -> str:
    if not values:
        return "(none)"
    return "\n".join([f"- {v.name} = {v.value} {v.unit}".rstrip() for v in values])


# =========================================================
# LLM helper: PID spec
# =========================================================
def _llm_make_pid_spec(
    client: OpenAI,
    spec: Spec,
    strategy: Strategy,
    model: DynamicsModel,
    revision_feedback: str = "",
) -> PIDDesignSpec:
    constraints_text = "\n".join(
        [f"- {c.name}: {c.value} {c.unit or ''} ({c.kind})" for c in spec.constraints]
    ) or "(none)"

    op_block = ""
    if model.operating_point is not None:
        op_block = (
            f"\nOperating point name: {model.operating_point.name}\n"
            f"Operating point states:\n{_named_values_to_lines(model.operating_point.state_values)}\n"
            f"Operating point inputs:\n{_named_values_to_lines(model.operating_point.input_values)}\n"
        )

    cr_block = ""
    if model.controller_ready is not None:
        cr_block += f"\nController-ready model class: {model.controller_ready.model_class}\n"
        if model.controller_ready.second_order is not None:
            so = model.controller_ready.second_order
            cr_block += (
                "Controller-ready second-order info:\n"
                f"- valid: {so.valid}\n"
                f"- form: {so.form}\n"
                f"- y_name: {so.y_name}\n"
                f"- ydot_name: {so.ydot_name}\n"
                f"- y_index_in_state: {so.y_index_in_state}\n"
                f"- ydot_index_in_state: {so.ydot_index_in_state}\n"
                f"- a1: {so.a1}\n"
                f"- a0: {so.a0}\n"
                f"- b0: {so.b0}\n"
            )

    revision_block = ""
    if revision_feedback.strip():
        revision_block = f"\nQC REVISION FEEDBACK:\n{revision_feedback}\n"

    system = (
        "You are a control engineer designing a model-based PID-family controller.\n"
        "Choose P / PI / PD / PID and implementation details based on the task and the provided structured model.\n"
        "Output MUST follow the PIDDesignSpec schema exactly.\n\n"
        "Rules:\n"
        "- The plant's controller-ready second-order local model has already been identified upstream when available.\n"
        "- Do not invent new plant coefficients here and do not re-derive a0, a1, or b0.\n"
        "- Your job here is only to choose the controller structure (P / PI / PD / PID) and implementation settings.\n"
        "- If QC revision feedback is provided, preserve the valid controller family choice when possible and prioritize repairing controller-package consistency, signal naming, operating-point annotation, and metadata completeness.\n"
        "- For local stabilization of a second-order mechanical system, PD is usually the preferred first choice unless strict steady-state error requirements justify integral action.\n"
        "- Add integral action only if steady-state error requirements are important and actuator saturation / windup concerns are manageable.\n"
        "- Use derivative on measurement by default.\n"
        "- Keep dt, settling time, overshoot, and actuator limit consistent with the scenario and defaults.\n"
        "- If actuator limit is unknown, you may use null.\n"
        "- Prefer the simplest controller that satisfies the intent.\n"
    )

    user = (
        f"Scenario: {spec.scenario}\n\n"
        f"Controller type = {strategy.family}\n"
        f"Design targets (defaults may be present): dt={spec.design_targets.dt_s}, "
        f"Ts={spec.design_targets.settle_time_s}, Mp={spec.design_targets.overshoot_pct}, "
        f"torque_limit={spec.design_targets.torque_limit_nm}\n\n"
        f"Constraints:\n{constraints_text}\n\n"
        f"Model name: {model.model_name}\n"
        f"Model states: {[v.name for v in model.states]}\n"
        f"Model inputs: {[v.name for v in model.inputs]}\n"
        f"Model outputs: {[v.name for v in model.outputs]}\n"
        f"ODE:\n- " + "\n- ".join(model.ode)
        + op_block
        + cr_block
        + revision_block
    )

    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=PIDDesignSpec,
    )

    return resp.output_parsed


# =========================================================
# LLM helper: LQR spec
# =========================================================
def _llm_make_lqr_spec(
    client: OpenAI,
    spec: Spec,
    model: DynamicsModel,
    revision_feedback: str = "",
) -> LQRSpec:
    constraints_text = "\n".join(
        [f"- {c.name}: {c.value} {c.unit or ''} ({c.kind})" for c in spec.constraints]
    ) or "(none)"

    op_block = "(none)"
    lin_block = "(none)"
    if model.operating_point is not None:
        op_block = (
            f"Operating point name: {model.operating_point.name}\n"
            f"State values:\n{_named_values_to_lines(model.operating_point.state_values)}\n"
            f"Input values:\n{_named_values_to_lines(model.operating_point.input_values)}"
        )
    if model.local_linear_model is not None:
        llm = model.local_linear_model
        lin_block = (
            f"valid_near: {llm.valid_near}\n"
            f"state_order: {llm.state_order}\n"
            f"input_order: {llm.input_order}\n"
            f"state_definition: {llm.state_definition}\n"
            f"input_definition: {llm.input_definition}\n"
            f"feedback_coordinates: {getattr(llm, 'feedback_coordinates', 'deviation')}\n"
            f"final_control_law_template: {getattr(llm, 'final_control_law_template', 'u_phys = u_eq + delta_u')}\n"
            f"A: {llm.A.data}\n"
            f"B: {llm.B.data}\n"
            f"C: {None if llm.C is None else llm.C.data}\n"
            f"D: {None if llm.D is None else llm.D.data}"
        )

    revision_block = ""
    if revision_feedback.strip():
        revision_block = f"\nQC REVISION FEEDBACK:\n{revision_feedback}\n"

    system = (
        "You are a control engineer preparing weighting choices for LQR synthesis.\n"
        "A controller-ready operating point and local linear model have already been identified upstream.\n"
        "Your job is to preserve that upstream structure and only choose a consistent LQR design package.\n"
        "Output MUST follow the schema exactly.\n\n"
        "Rules:\n"
        "- Treat the upstream operating point as the source of truth for physical equilibrium values.\n"
        "- Treat the upstream local linear model as the source of truth for deviation-coordinate definitions and for A,B matrices.\n"
        "- Do not invent a different operating point.\n"
        "- Do not alter state_order, input_order, deviation_state_definition, deviation_input_definition, or final_control_law_template unless there is an obvious inconsistency.\n"
        "- Reuse the provided A and B matrices exactly whenever they are already available and dimensionally consistent.\n"
        "- Your main task is to choose numeric Q and R matrices and provide concise notes explaining the weighting rationale.\n"
        "- Q and R must be dimensionally consistent with the provided A and B.\n"
        "- Prefer principled heuristics: penalize configuration/state error more than rate error when appropriate, and penalize excessive control effort.\n"
        "- Preserve uses_deviation_variables if the upstream model uses deviation coordinates.\n"
        "- physical_state_eq and physical_input_eq must come directly from the upstream operating point.\n"
        "- Do not restate the final controller gain K here; K will be solved deterministically downstream.\n"
        "- If QC revision feedback is provided, preserve the valid upstream operating point and local model, and prioritize repairing design-package consistency, metadata completeness, and interface alignment.\n"
    )

    user = (
        f"Scenario: {spec.scenario}\n\n"
        f"Design targets (defaults may be present): dt={spec.design_targets.dt_s}, "
        f"Ts={spec.design_targets.settle_time_s}, Mp={spec.design_targets.overshoot_pct}\n\n"
        f"Constraints:\n{constraints_text}\n\n"
        f"Model name: {model.model_name}\n"
        f"Model states: {[v.name for v in model.states]}\n"
        f"Model input: {[v.name for v in model.inputs]}\n"
        f"Model outputs: {[v.name for v in model.outputs]}\n"
        f"ODE:\n- " + "\n- ".join(model.ode) + "\n\n"
        f"Structured operating point:\n{op_block}\n\n"
        f"Structured local linear model:\n{lin_block}\n"
        + revision_block
    )

    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=LQRSpec,
    )

    return resp.output_parsed


# =========================================================
# Deterministic helper: validate + solve + build LQR controller
# =========================================================
def _build_lqr_controller_from_spec(model: DynamicsModel, lqr: LQRSpec) -> ControllerDesign:
    n = lqr.A.rows
    if lqr.A.cols != n:
        raise ValueError("LQRSpec A must be square.")
    if lqr.B.rows != n:
        raise ValueError("LQRSpec B row count must match A.")
    if lqr.Q.rows != n or lqr.Q.cols != n:
        raise ValueError("LQRSpec Q shape mismatch.")

    m = lqr.B.cols
    if lqr.R.rows != m or lqr.R.cols != m:
        raise ValueError("LQRSpec R shape mismatch.")
    if len(lqr.state_order) != n:
        raise ValueError("LQRSpec state_order length must match A dimension.")

    input_order = lqr.input_order or ([lqr.input_name] if lqr.input_name else [])
    if not input_order:
        if model.local_linear_model is not None and model.local_linear_model.input_order:
            input_order = model.local_linear_model.input_order
        else:
            input_order = [v.name for v in model.inputs]

    if len(input_order) != m:
        raise ValueError("LQRSpec input_order length must match B/R input dimension.")

    physical_input_names = [v.name for v in model.inputs]
    local_input_names = []
    if model.local_linear_model is not None and model.local_linear_model.input_order:
        local_input_names = list(model.local_linear_model.input_order)

    for nm in input_order:
        if nm not in physical_input_names and nm not in local_input_names:
            raise ValueError(
                f"LQRSpec input '{nm}' not found in either model.inputs or model.local_linear_model.input_order."
            )

    K = solve_lqr(
        A=lqr.A.data,
        B=lqr.B.data,
        Q=lqr.Q.data,
        R=lqr.R.data,
        discrete=False,
    )

    law = [
        f"{lqr.deviation_state_definition}",
        f"{lqr.deviation_input_definition}",
        f"delta_u = -K delta_x,  K = {K.tolist()}",
        f"{lqr.final_control_law_template}",
    ]

    params = [
        Parameter(
            name="K",
            meaning=f"LQR feedback gain for state order {lqr.state_order}",
            unit="(depends on input/state units)",
            value=str(K.tolist()),
            notes=f"Computed from {'DARE' if lqr.discrete else 'CARE'}; operating point {lqr.operating_point_name}",
        )
    ]
    for v in lqr.physical_state_eq:
        params.append(
            Parameter(
                name=f"{v.name}_eq",
                meaning=f"operating-point value for state {v.name}",
                unit=v.unit,
                value=str(v.value),
                notes=v.notes,
            )
        )
    for v in lqr.physical_input_eq:
        params.append(
            Parameter(
                name=f"{v.name}_eq",
                meaning=f"operating-point value for input {v.name}",
                unit=v.unit,
                value=str(v.value),
                notes=v.notes,
            )
        )

    controller = ControllerDesign(
        controller_type="LQR",
        controller_name="local_lqr_controller",
        required_signals=lqr.state_order,
        control_law=law,
        parameters=params,
        assumptions=[
            "Valid near the linearization point (local stabilization).",
            "Model linearization matches the implemented deviation-state definitions.",
        ],
        implementation_notes=[
            f"Operating point: {lqr.operating_point_name}",
            f"Uses deviation variables: {lqr.deviation_state_definition}; {lqr.deviation_input_definition}",
            "Add input saturation if actuator limits exist.",
            "If using discrete control in simulation, set discrete=True and provide dt_s.",
        ] + (lqr.notes or []),
        designed_for_operating_point=lqr.operating_point_name,
        designed_for_linear_model=lqr.operating_point_name,
        uses_deviation_variables=lqr.uses_deviation_variables,
        state_order=lqr.state_order,
        input_order=input_order,
        physical_state_eq=lqr.physical_state_eq,
        physical_input_eq=lqr.physical_input_eq,
    )

    return controller


# =========================================================
# Agent
# =========================================================
def control_agent(state: dict) -> dict:
    spec: Spec = state["spec"]
    strategy: Strategy = state["strategy"]
    model: DynamicsModel = state["model"]

    revision_feedback = state.get("revision_requests", {}).get("control", "")
    family = strategy.family.upper().strip()

    if family in {"P", "PI", "PD", "PID"}:
        try:
            pid_spec = _llm_make_pid_spec(
                client,
                spec,
                strategy,
                model,
                revision_feedback=revision_feedback,
            )
            controller = solve_pid_from_model(model, pid_spec)

            # Fill package metadata directly from the structured model.
            if model.operating_point is not None:
                controller.designed_for_operating_point = model.operating_point.name
                controller.designed_for_linear_model = model.operating_point.name
                controller.physical_state_eq = model.operating_point.state_values
                controller.physical_input_eq = model.operating_point.input_values

            if model.local_linear_model is not None:
                controller.state_order = model.local_linear_model.state_order
                controller.input_order = model.local_linear_model.input_order
                feedback_coordinates = getattr(model.local_linear_model, "feedback_coordinates", "")
                controller.uses_deviation_variables = (feedback_coordinates == "deviation")

            # Align required signals with model/controller-ready naming.
            if model.controller_ready is not None and model.controller_ready.second_order is not None:
                so = model.controller_ready.second_order
                if getattr(so, "valid", False):
                    controller.required_signals = [so.y_name, so.ydot_name]

            # Mark Ki clearly as unused for PD if present as zero metadata.
            if family == "PD":
                for p in controller.parameters:
                    if p.name == "Ki" and str(p.value) in {"0", "0.0", "0.000000"}:
                        existing = p.notes or ""
                        if "Unused for PD" not in existing:
                            p.notes = (existing + " Unused for PD.").strip()

            if "revision_requests" in state and isinstance(state["revision_requests"], dict):
                state["revision_requests"]["control"] = ""

            state["controller"] = controller
            state.setdefault("qc", []).append(
                QCReport(
                    step="control",
                    reviewer="control_agent_self_report",
                    passed=True,
                    verdict="pass",
                    retry_target="none",
                    severity="info",
                    confidence=0.90,
                    summary=f"{controller.controller_type} controller synthesized successfully.",
                    fail_reasons=[],
                    evidence=[
                        f"family={family}",
                        f"controller_type={controller.controller_type}",
                        "PID-family numeric gains computed successfully.",
                    ],
                    internal_feedback="",
                    user_feedback="",
                )
            )
            return state

        except Exception as e:
            controller = ControllerDesign(
                controller_type=family,
                required_signals=[v.name for v in model.outputs],
                control_law=[f"{model.inputs[0].name} = <{family} controller tuning failed>"],
                implementation_notes=[f"PID auto-tuning failed: {str(e)}"],
            )
            state["controller"] = controller
            state.setdefault("qc", []).append(
                QCReport(
                    step="control",
                    reviewer="control_agent_self_report",
                    passed=False,
                    verdict="retry",
                    retry_target="control",
                    severity="error",
                    confidence=0.95,
                    summary="PID-family controller synthesis failed.",
                    fail_reasons=[str(e)],
                    evidence=[
                        f"family={family}",
                        "Exception raised during PID spec generation or deterministic solve.",
                    ],
                    internal_feedback=(
                        "Re-run control synthesis for the same model and strategy. "
                        "Inspect PIDDesignSpec generation and solve_pid_from_model inputs."
                    ),
                    user_feedback="",
                )
            )
            return state

    if family == "LQR":
        try:
            lqr_spec = _llm_make_lqr_spec(
                client,
                spec,
                model,
                revision_feedback=revision_feedback,
            )
            controller = _build_lqr_controller_from_spec(model, lqr_spec)

            if "revision_requests" in state and isinstance(state["revision_requests"], dict):
                state["revision_requests"]["control"] = ""

            state["controller"] = controller
            state.setdefault("qc", []).append(
                QCReport(
                    step="control",
                    reviewer="control_agent_self_report",
                    passed=True,
                    verdict="pass",
                    retry_target="none",
                    severity="info",
                    confidence=0.90,
                    summary="LQR controller synthesized successfully.",
                    fail_reasons=[],
                    evidence=[
                        "LQR spec generated successfully.",
                        "Numeric feedback gain K computed successfully.",
                    ],
                    internal_feedback="",
                    user_feedback="",
                )
            )
            return state

        except Exception as e:
            controller = ControllerDesign(
                controller_type="LQR",
                required_signals=[v.name for v in model.outputs],
                control_law=[f"{model.inputs[0].name} = <LQR design failed>"],
                implementation_notes=[f"LQR auto-design failed: {str(e)}"],
            )
            state["controller"] = controller
            state.setdefault("qc", []).append(
                QCReport(
                    step="control",
                    reviewer="control_agent_self_report",
                    passed=False,
                    verdict="retry",
                    retry_target="control",
                    severity="error",
                    confidence=0.95,
                    summary="LQR controller synthesis failed.",
                    fail_reasons=[str(e)],
                    evidence=[
                        "Exception raised during LQR spec generation or deterministic solve.",
                    ],
                    internal_feedback=(
                        "Re-run control synthesis for the same model. "
                        "Inspect LQRSpec dimensions, A/B/Q/R consistency, and model-controller interface."
                    ),
                    user_feedback="",
                )
            )
            return state

    controller = ControllerDesign(
        controller_type=family,
        required_signals=[v.name for v in model.outputs],
        control_law=[f"{model.inputs[0].name} = <unsupported controller family>"],
        implementation_notes=[f"Unsupported controller family in control_agent: {family}"],
    )
    state["controller"] = controller

    state.setdefault("qc", []).append(
        QCReport(
            step="control",
            reviewer="control_agent_self_report",
            passed=False,
            verdict="stop",
            retry_target="user",
            severity="error",
            confidence=0.99,
            summary=f"Unsupported controller family: {family}.",
            fail_reasons=[f"Unsupported controller family in control_agent: {family}"],
            evidence=[
                f"strategy.family={family}",
            ],
            internal_feedback=(
                "Do not retry control synthesis until strategy selection is changed to a supported family."
            ),
            user_feedback=(
                f"The requested controller family '{family}' is not currently supported by this demo."
            )
        )
    )

    return state
