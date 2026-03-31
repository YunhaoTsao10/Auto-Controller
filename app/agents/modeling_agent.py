from app.config import client, OPENAI_MODEL
from app.schemas import Spec, DynamicsModel, QCReport


def modeling_agent(state: dict) -> dict:
    spec: Spec = state["spec"]

    constraints_text = "\n".join(
        [f"- {c.name}: {c.value} {c.unit or ''} ({c.kind})" for c in spec.constraints]
    ) or "(none)"

    strategy = state.get("strategy")
    family = getattr(strategy, "family", None) if strategy else None
    family_text = family or "unknown"

    system = (
        "You are a robotics/controls modeling engineer.\n"
        "Given a control scenario, produce a compact but usable dynamics model.\n"
        "Output must follow the provided schema exactly.\n\n"
        "General rules:\n"
        "- Prefer continuous-time physics-based modeling when appropriate.\n"
        "- Use standard minimal state definitions.\n"
        "- Include parameter list with SI units.\n"
        "- Be consistent with symbols: if you use omega, define it as a state variable.\n"
        "- If the scenario includes an explicit plant model (transfer function, ODE, or state-space), preserve it and do not replace it with a different physical system.\n"
        "- Fill BOTH the structured fields (dynamics, operating_point, local_linear_model) and the backward-compatible plain-text fields (ode, state_space) whenever possible.\n"
        "- Never hide operating-point information only inside free-form notes. Put it in operating_point.\n"
        "- If a local controller is intended, explicitly provide the physical operating point values for states and inputs.\n"
        "- If the plant is nonlinear and the task is local stabilization or local tracking, provide both the original nonlinear model and the local linearized model around the requested operating point.\n"
        "- For a nonzero operating point, the local linear model must be written in deviation variables around that operating point.\n"
        "- In local_linear_model, A and B must correspond to deviation dynamics: delta_x_dot = A delta_x + B delta_u.\n"
        "- Explicitly define deviation variables in local_linear_model.state_definition and input_definition.\n"
        "- local_linear_model.valid_near should match operating_point.name.\n"
        "- state_order and input_order in local_linear_model must match the actual variable names.\n"
        "- If output matrices are obvious, provide C and D too.\n"
        "- If numeric coefficients can be derived from the scenario, do not output 'unknown'.\n"
        "Additional requirements when the selected controller family is LQR:\n"
        "- The model must be suitable for local state-feedback design around a specific operating point.\n"
        "- The declared operating_point must contain explicit physical state and input values.\n"
        "- The local linear model must be a first-order linearization of the declared nonlinear dynamics around that operating point.\n"
        "- The linearization must use deviation coordinates, and these coordinates must be explicitly defined.\n"
        "- The state_order and input_order must exactly match the declared variable names and the matrix dimensions.\n"
        "- If the operating point is claimed to be an equilibrium or trim point, ensure it is dynamically consistent with the nonlinear dynamics.\n"
        "- Do not provide a generic origin-based linearization when the task specifies a nonzero operating point.\n"
        "- Include enough structure so that a downstream LQR solver can apply u = u_op + delta_u with delta_u = -K delta_x without ambiguity.\n"
        "Controller-family-specific guidance:\n"
        "- If the controller family is PID/PI/PD/P and the plant can be represented locally as a second-order SISO system, populate controller_ready.\n"
        "- Set controller_ready.model_class = 'second_order_siso'.\n"
        "- Populate controller_ready.second_order with:\n"
        "- valid = true\n"
        "- form = 'delta_y_ddot + a1*delta_y_dot + a0*delta_y = b0*delta_u'\n"
        "- y_name\n"
        "- ydot_name\n"
        "- y_index_in_state\n"
        "- ydot_index_in_state\n"
        "- a1, a0, b0\n"
        "- These coefficients must correspond to the local deviation model around the declared operating point.\n"
        "- The indices in controller_ready.second_order must match local_linear_model.state_order exactly.\n"
        "- If a bias or equilibrium input is required to hold the operating point, include it explicitly in operating_point.input_values.\n"
        "- Do not rely on free-form notes to store controller-ready coefficients.\n"
        "- Those canonical coefficients must correspond to the local deviation model around the detected operating point.\n"
        "- If the controller family is LQR, provide explicit numeric local_linear_model A, B, C, D matrices and a clear operating_point.\n"
        "- If a bias or equilibrium input is required to hold the operating point, include it both in operating_point.input_values and in model.parameters as a named parameter such as u_eq when appropriate.\n"
        "- Also include operating-point state values in model.parameters when useful for downstream solvers (theta_eq, x_eq, etc.), but operating_point remains the primary source of truth.\n\n"
        "Self-consistency checks to satisfy before answering:\n"
        "- The operating point and the local linear model must refer to the same point.\n"
        "- The state/input names used in matrices must match the declared state/input names.\n"
        "- If the operating point is nonzero, do not silently use an origin small-signal model instead.\n"
    )

    user = (
        f"Selected controller family: {family_text}\n\n"
        f"Scenario:\n{spec.scenario}\n\n"
        f"Constraints:\n{constraints_text}\n\n"
        "If ambiguous, choose the simplest reasonable model and list assumptions."
    )

    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=DynamicsModel,
    )
    model: DynamicsModel = resp.output_parsed

    state["model"] = model
    state.setdefault("qc", []).append(QCReport(step="modeling", passed=True, notes=model.model_name))
    return state
