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
def _llm_make_pid_spec(client: OpenAI, spec: Spec, strategy: Strategy, model: DynamicsModel) -> PIDDesignSpec:
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

    system = (
        "You are a control engineer designing a model-based PID family controller.\n"
        "Choose P / PI / PD / PID and implementation details based on the task and model.\n"
        "Output MUST follow the schema exactly.\n\n"
        "Rules:\n"
        "- For local stabilization of a second-order mechanical system (e.g., pendulum near equilibrium), PD is often preferred first unless a specific type of control is required.\n"
        "- Add integral action only if steady-state error requirement is strict and actuator saturation can be handled.\n"
        "- Use derivative on measurement by default.\n"
        "- Keep dt, settling time, overshoot consistent with user intent or defaults.\n"
        "- If actuator limit is unknown, you may use null.\n"
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
def _llm_make_lqr_spec(client: OpenAI, spec: Spec, model: DynamicsModel) -> LQRSpec:
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
            f"A: {llm.A.data}\n"
            f"B: {llm.B.data}\n"
            f"C: {None if llm.C is None else llm.C.data}\n"
            f"D: {None if llm.D is None else llm.D.data}"
        )

    system = (
        "You are a control engineer. We will implement LQR.\n"
        "Given the dynamics model, produce an LQR design package using the provided local linear model whenever available.\n"
        "Output MUST follow the schema exactly.\n\n"
        "Rules:\n"
        "- Reuse the operating point and local linear model already identified in the model instead of inventing a different one.\n"
        "- Provide numeric A,B,Q,R and a clear state/input order.\n"
        "- state_order must match the linear-model state order.\n"
        "- input_order must match the model input names.\n"
        "- If you use deviation variables, they must be consistent with the operating point.\n"
        "- Set physical_state_eq and physical_input_eq from the operating point values.\n"
        "- Choose Q,R using a principled heuristic (e.g., penalize configuration error more than velocity; penalize control effort).\n"
        "- Keep dimensions consistent.\n"
        "- If a local linear model is already available in the model, do not alter A or B unless there is an obvious inconsistency.\n"
        "- Notes should mention the operating point and weighting rationale.\n"
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
        input_order = [v.name for v in model.inputs]
    if len(input_order) != m:
        raise ValueError("LQRSpec input_order length must match B/R input dimension.")
    for nm in input_order:
        if nm not in [v.name for v in model.inputs]:
            raise ValueError(f"LQRSpec input '{nm}' not found in model inputs.")

    K = solve_lqr(
        A=lqr.A.data,
        B=lqr.B.data,
        Q=lqr.Q.data,
        R=lqr.R.data,
        discrete=False,
    )

    input_name = input_order[0] if len(input_order) == 1 else "u"
    if K.shape[0] == 1:
        x_expr = " + ".join([f"{K[0, i]:.6f}*{name}" for i, name in enumerate(lqr.state_order)])
        law = [
            f"delta_x = [{', '.join(lqr.state_order)}]^T (deviation-state order)",
            f"delta_{input_name} = -({x_expr})",
            f"{input_name} = {input_name}_eq + delta_{input_name}",
        ]
    else:
        law = [
            f"delta_x = [{', '.join(lqr.state_order)}]^T (deviation-state order)",
            f"delta_u = -K delta_x,  K = {K.tolist()}",
            f"u = u_eq + delta_u",
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

    family = strategy.family.upper().strip()

    if family in {"P", "PI", "PD", "PID"}:
        try:
            pid_spec = _llm_make_pid_spec(client, spec, strategy, model)
            controller = solve_pid_from_model(model, pid_spec)

            state["controller"] = controller
            state.setdefault("qc", []).append(
                QCReport(
                    step="control(PID+solve)",
                    passed=True,
                    notes=f"{controller.controller_type} numeric gains computed",
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
                QCReport(step="control(PID+solve)", passed=False, notes=str(e))
            )
            return state

    if family == "LQR":
        try:
            lqr_spec = _llm_make_lqr_spec(client, spec, model)
            controller = _build_lqr_controller_from_spec(model, lqr_spec)

            state["controller"] = controller
            state.setdefault("qc", []).append(
                QCReport(step="control(LQR+solve)", passed=True, notes="numeric K computed")
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
                QCReport(step="control(LQR+solve)", passed=False, notes=str(e))
            )
            return state

    controller = ControllerDesign(
        controller_type=family,
        required_signals=[v.name for v in model.outputs],
        control_law=[f"{model.inputs[0].name} = <unsupported controller family>"],
        implementation_notes=[f"Unsupported controller family in control_agent: {family}"],
    )
    state["controller"] = controller
    state.setdefault("qc", []).append(QCReport(step="control", passed=False, notes=f"unsupported family {family}"))
    return state
