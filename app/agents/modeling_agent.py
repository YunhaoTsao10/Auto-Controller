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

    revision_feedback = state.get("revision_requests", {}).get("model", "")

    system = (
        "You are a robotics/controls modeling engineer.\n"
        "Given a control scenario, produce a compact but usable dynamics model.\n"
        "Output must follow the provided schema exactly.\n\n"

        "General rules:\n"
        "- Prefer continuous-time physics-based modeling when appropriate.\n"
        "- Use standard minimal state definitions.\n"
        "- Include parameter list with SI units.\n"
        "- Be consistent with symbols: if you use omega, define it as a state variable.\n"
        "- If the scenario includes an explicit plant model (transfer function, ODE, state-space, XML, URDF, or other descriptive model file), preserve and interpret that plant information rather than replacing it with an unrelated toy system.\n"
        "- Fill BOTH the structured fields (dynamics, operating_point, local_linear_model) and the backward-compatible plain-text fields (ode, state_space) whenever possible.\n"
        "- Never hide operating-point information only inside free-form notes. Put it in operating_point.\n"
        "- If a local controller is intended, explicitly provide the physical operating point values for states and inputs.\n"
        "- If the plant is nonlinear and the task is local stabilization or local tracking, provide both the original nonlinear model and the local linearized model around the requested operating point.\n"
        "- Prefer the smallest model that is still faithful to the requested control objective and the available plant description.\n"
        "- Do not collapse a clearly multi-axis or multi-actuator system into an unrelated single-axis toy model when the task explicitly requests multi-state or multi-input control.\n"
        "- If the uploaded model description clearly exposes multiple actuators, free-body motion, or multi-axis inertia, preserve that structure unless the user explicitly asks for a reduced-order abstraction.\n"
        "- For a nonzero operating point, the local linear model must be written in deviation variables around that operating point.\n"
        "- In local_linear_model, A and B must correspond to deviation dynamics: delta_x_dot = A delta_x + B delta_u.\n"
        "- Explicitly define deviation variables in local_linear_model.state_definition and input_definition.\n"
        "- local_linear_model.valid_near should match operating_point.name.\n"
        "- state_order and input_order in local_linear_model must match the actual declared variable names and matrix dimensions exactly.\n"
        "- When providing a local linear model, treat operating_point as the source of truth for equilibrium values and local_linear_model as the source of truth for deviation-coordinate definitions.\n"
        "- Do not duplicate equilibrium information elsewhere in inconsistent forms.\n"
        "- If local_linear_model.state_definition uses deviation coordinates, it must be consistent with operating_point.state_values.\n"
        "- If local_linear_model.input_definition uses deviation coordinates, it must be consistent with operating_point.input_values.\n"
        "- The local linear model must make it unambiguous how a downstream controller-synthesis agent reconstructs physical input from feedback-space input.\n"
        "- Use local_linear_model.final_control_law_template to state only that mapping explicitly, for example: u_phys = u_eq + delta_u.\n"
        "- Do not encode controller-family-specific gains or final feedback formulas inside the model unless the schema explicitly asks for them.\n"
        "- If output matrices are obvious, provide C and D too.\n"
        "- If numeric coefficients can be derived from the scenario, do not output 'unknown'.\n\n"

        "Controller-family-specific guidance:\n"
        "- If the selected controller family is LQR, provide a controller-ready local linear model around the declared operating point.\n"
        "- For LQR-oriented modeling, make state order, input order, equilibrium values, and deviation-coordinate conventions explicit enough for downstream state-feedback synthesis.\n"
        "- If the selected controller family is PID/PI/PD/P and the plant can be represented locally as a second-order SISO system, populate controller_ready.\n"
        "- Set controller_ready.model_class = 'second_order_siso'.\n"
        "- Populate controller_ready.second_order with coefficients that correspond to the local deviation model around the declared operating point.\n"
        "- The indices in controller_ready.second_order must match local_linear_model.state_order exactly.\n"
        "- If a bias or equilibrium input is required to hold the operating point, include it explicitly in operating_point.input_values.\n"
        "- Do not rely on free-form notes to store controller-ready coefficients.\n\n"

        "Self-consistency checks to satisfy before answering:\n"
        "- The operating point and the local linear model must refer to the same point.\n"
        "- The state/input names used in matrices must match the declared state/input names.\n"
        "- If the operating point is nonzero, do not silently use an origin small-signal model instead.\n"
        "- If local_linear_model is present, state_definition, input_definition, and final_control_law_template must be mutually consistent.\n"
    )

    user = (
        f"Selected controller family: {family_text}\n\n"
        f"Scenario:\n{spec.scenario}\n\n"
        f"Constraints:\n{constraints_text}\n\n"
        "If ambiguous, choose the simplest reasonable model and list assumptions."
    )

    if revision_feedback:
        user += f"\nQC revision feedback:\n{revision_feedback}\n"

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
    state.setdefault("qc", []).append(
        QCReport(
            step="model",
            reviewer="modeling_agent_self_report",
            passed=True,
            verdict="pass",
            retry_target="none",
            severity="info",
            confidence=0.90,
            summary=f"Model generated: {model.model_name}",
            fail_reasons=[],
            evidence=[
                f"model_name={model.model_name}",
            ],
            internal_feedback="",
            user_feedback="",
        )
    )
    return state
