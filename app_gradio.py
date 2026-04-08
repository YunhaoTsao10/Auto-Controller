from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from app.graph import build_app


# -----------------------------------------------------------------------------
# Backend helpers
# -----------------------------------------------------------------------------

def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _build_snapshot(out: dict[str, Any]) -> dict[str, Any]:
    return {
        "spec": _dump(out.get("spec")),
        "strategy": _dump(out.get("strategy")),
        "model": _dump(out.get("model")),
        "controller": _dump(out.get("controller")),
        "qc": [_dump(q) for q in out.get("qc", [])] if out.get("qc") else None,
    }


def run_design(user_instruction: str, urdf_file: str | None):
    if not user_instruction or not user_instruction.strip():
        raise gr.Error("Please provide a control-design instruction.")

    app = build_app()

    xml_text = None
    if urdf_file:
        xml_text = Path(urdf_file).read_text(encoding="utf-8", errors="ignore")

        state = {
            "raw_request": user_instruction.strip() + "\n\nUploaded robot.xml content:\n" + xml_text,
            "urdf": urdf_file if urdf_file else None,
        }
    
    else:
        state = {
            "raw_request": user_instruction.strip(),
            "urdf": None,
        }

    out = app.invoke(state)
    snapshot = _build_snapshot(out)

    model_md = render_model_markdown(snapshot)
    controller_md = render_controller_markdown(snapshot)

    snapshot_path = Path("controller_snapshot.json")
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    return snapshot, model_md, controller_md


# -----------------------------------------------------------------------------
# Markdown renderers
# -----------------------------------------------------------------------------

def _named_value_lines(values: list[dict[str, Any]] | None) -> list[str]:
    if not values:
        return ["- None"]

    lines: list[str] = []
    for item in values:
        name = item.get("name", "?")
        value = item.get("value", "?")
        unit = item.get("unit", "")
        notes = item.get("notes", "")
        suffix = f" {unit}" if unit else ""
        note = f" — {notes}" if notes else ""
        lines.append(f"- `{name}` = {value}{suffix}{note}")
    return lines


def render_model_markdown(snapshot: dict[str, Any]) -> str:
    model = snapshot.get("model")
    if not model:
        return "## Dynamics Model\n\nNo model generated."

    states = ", ".join(f"`{s['name']}`" for s in model.get("states", [])) or "None"
    inputs = ", ".join(f"`{u['name']}`" for u in model.get("inputs", [])) or "None"
    outputs = ", ".join(f"`{y['name']}`" for y in model.get("outputs", [])) or "None"

    lines: list[str] = [
        "## Dynamics Model",
        "",
        f"**Model name:** {model.get('model_name', 'Unknown')}",
        f"**Model type:** {model.get('model_type', 'Unknown')}",
        "",
        f"**States:** {states}",
        f"**Inputs:** {inputs}",
        f"**Outputs:** {outputs}",
    ]

    params = model.get("parameters") or []
    if params:
        lines.extend(["", "### Parameters"])
        for p in params:
            unit = f" [{p.get('unit', '')}]" if p.get("unit") else ""
            notes = f" — {p.get('notes')}" if p.get("notes") else ""
            lines.append(f"- `{p.get('name', '?')}` = {p.get('value', '?')}{unit}{notes}")

    dynamics = model.get("dynamics") or []
    ode = model.get("ode") or []
    if dynamics or ode:
        lines.extend(["", "### Dynamics"])
        if dynamics:
            for eq in dynamics:
                lhs = eq.get("lhs", "?")
                rhs = eq.get("rhs", "?")
                lines.append(f"- `{lhs} = {rhs}`")
        else:
            for eq in ode:
                lines.append(f"- `{eq}`")

    operating_point = model.get("operating_point")
    if operating_point:
        lines.extend([
            "",
            "### Operating Point",
            f"**Name:** {operating_point.get('name', 'Unknown')}",
            f"**Kind:** {operating_point.get('kind', 'Unknown')}",
            "",
            "**State values**",
            *_named_value_lines(operating_point.get("state_values")),
            "",
            "**Input values**",
            *_named_value_lines(operating_point.get("input_values")),
        ])

    linear = model.get("local_linear_model")
    if linear:
        lines.extend([
            "",
            "### Local Linear Model",
            f"**Valid near:** {linear.get('valid_near', 'Unknown')}",
            f"**State definition:** `{linear.get('state_definition', '')}`",
            f"**Input definition:** `{linear.get('input_definition', '')}`",
            f"**State order:** {', '.join([f'`{x}`' for x in linear.get('state_order', [])]) or 'None'}",
            f"**Input order:** {', '.join([f'`{u}`' for u in linear.get('input_order', [])]) or 'None'}",
        ])

        A = linear.get("A", {}).get("data") if linear.get("A") else None
        B = linear.get("B", {}).get("data") if linear.get("B") else None
        if A is not None or B is not None:
            lines.extend(["", "**Matrices**"])
            if A is not None:
                lines.append(f"- `A = {A}`")
            if B is not None:
                lines.append(f"- `B = {B}`")

        lines.extend([
            "",
            "### Canonical Local Form",
            r"$$\delta \dot{x} = A\,\delta x + B\,\delta u$$",
        ])

    controller_ready = model.get("controller_ready")
    if controller_ready:
        lines.extend([
            "",
            "### Controller-Ready View",
            f"**Model class:** {controller_ready.get('model_class', 'Unknown')}",
        ])

        so = controller_ready.get("second_order")
        if so:
            lines.extend([
                "",
                "**Second-order canonical form**",
                f"- `valid` = {so.get('valid')}",
                f"- `form` = `{so.get('form', '')}`",
                f"- `y_name` = `{so.get('y_name', '?')}`",
                f"- `ydot_name` = `{so.get('ydot_name', '?')}`",
                f"- `y_index_in_state` = {so.get('y_index_in_state', '?')}",
                f"- `ydot_index_in_state` = {so.get('ydot_index_in_state', '?')}",
                f"- `a1` = {so.get('a1', '?')}",
                f"- `a0` = {so.get('a0', '?')}",
                f"- `b0` = {so.get('b0', '?')}",
                "",
                "### Canonical Second-Order Form",
                r"$$\delta \ddot{y} + a_1 \delta \dot{y} + a_0 \delta y = b_0 \delta u$$",
            ])

        cr_notes = controller_ready.get("notes") or []
        if cr_notes:
            lines.extend(["", "**Controller-ready notes**"])
            for item in cr_notes:
                lines.append(f"- {item}")

    assumptions = model.get("assumptions") or []
    if assumptions:
        lines.extend(["", "### Assumptions"])
        for item in assumptions:
            lines.append(f"- {item}")

    notes = model.get("notes") or []
    if notes:
        lines.extend(["", "### Notes"])
        for item in notes:
            lines.append(f"- {item}")

    return "\n".join(lines)

def render_controller_markdown(snapshot: dict[str, Any]) -> str:
    controller = snapshot.get("controller")
    strategy = snapshot.get("strategy") or {}

    if not controller:
        return "## Controller Design\n\nNo controller generated."

    lines: list[str] = [
        "## Controller Design",
        "",
        f"**Chosen family:** {strategy.get('family', 'Unknown')}",
        f"**Controller type:** {controller.get('controller_type', 'Unknown')}",
    ]

    if strategy.get("reason"):
        lines.extend(["", f"**Rationale:** {strategy['reason']}"])

    required_signals = controller.get("required_signals") or []
    if required_signals:
        lines.extend(["", "### Required Signals"])
        for sig in required_signals:
            lines.append(f"- `{sig}`")

    law = controller.get("control_law") or []
    if law:
        lines.extend(["", "### Control Law"])
        for item in law:
            lines.append(f"- `{item}`")

        law_text = " ".join(law).lower()
        ctrl_type = controller.get("controller_type", "").upper()

        if ctrl_type == "LQR":
            lines.extend([
                "",
                "### Canonical Law",
                r"$$u = u_{eq} - K(x - x_{eq})$$",
            ])
        elif ctrl_type == "PD":
            lines.extend([
                "",
                "### Canonical Law",
                r"$$u = u_{eq} + K_p e - K_d \dot{y}$$",
            ])
        elif ctrl_type in {"PI", "PID"} or "integral" in law_text:
            lines.extend([
                "",
                "### Canonical Law",
                r"$$u = u_{eq} + K_p e + K_i \int e\,dt - K_d \dot{y}$$",
            ])

    params = controller.get("parameters") or []
    if params:
        lines.extend(["", "### Parameters"])
        for p in params:
            unit = f" [{p.get('unit', '')}]" if p.get("unit") else ""
            notes = f" — {p.get('notes')}" if p.get("notes") else ""
            lines.append(f"- `{p.get('name', '?')}` = {p.get('value', '?')}{unit}{notes}")

    assumptions = controller.get("assumptions") or []
    if assumptions:
        lines.extend(["", "### Assumptions"])
        for item in assumptions:
            lines.append(f"- {item}")

    implementation_notes = controller.get("implementation_notes") or []
    if implementation_notes:
        lines.extend(["", "### Implementation Notes"])
        for item in implementation_notes:
            lines.append(f"- {item}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

DEFAULT_PROMPT = (
    "Please model a simple pendulum and design a local PD stabilizing controller around "
    "theta = 2.5 rad. Use m = 1 kg, L = 1 m, g = 9.81 m/s^2, and damping b = 0.05. "
    "Target less than 10% overshoot and settling time around 2 seconds."
)

# Please model a mass-spring-damper system and design a local stabilizing PD controller around x = 1.0 m.
# Use m = 1.0 kg, c = 0.4 N·s/m, k = 4.0 N/m.
# Target less than 10% overshoot and settling time around 2 seconds.

# The uploaded robot.xml describes a free-floating space robot in MuJoCo.
# Task:
# Design a continuous-time LQR controller for point-to-point motion from a fixed initial state to a fixed goal state.
# Requirements:
# 1. Do not reduce the robot to a single-axis model.
# 2. Use a local linear state-space model suitable for LQR.
# 3. The state should include at least position, velocity, attitude, and angular velocity.
# 4. The control inputs should correspond to the robot actuators in the XML.
# 5. Reuse the uploaded XML content to infer the robot structure, inertia, and available actuators.
# 6. The result must provide:
#    - operating point
#    - state order
#    - input order
#    - A, B, Q, R matrices
#    - equilibrium state x_eq
#    - equilibrium input u_eq
#    - LQR feedback law u = u_eq - K(x - x_eq)
# Control objective:
# Move the robot from an initial waypoint to a target waypoint in free space while stabilizing attitude.
# Initial desired state:
# position = [x0, y0, z0]
# velocity = [0, 0, 0]
# quaternion = [1, 0, 0, 0]
# angular_velocity = [0, 0, 0]
# Target desired state:
# position = [xg, yg, zg]
# velocity = [0, 0, 0]
# quaternion = [1, 0, 0, 0]
# angular_velocity = [0, 0, 0]
# Important:
# - Treat this as a multi-input multi-state spacecraft control problem.
# - Do not collapse it into a SISO second-order template.
# - If needed, make clearly stated small-angle and local-linearity assumptions near the target state.



with gr.Blocks(title="Agentic Control Design Demo") as demo:
    gr.Markdown(
        "# Agentic Control Design Demo\n"
        "Upload an optional URDF and describe the control task in natural language.\n"
        "The app returns a structured model, a controller design, and the raw JSON snapshot."
    )

    with gr.Row():
        with gr.Column(scale=4):
            instruction = gr.Textbox(
                label="User Instruction",
                lines=10,
                value=DEFAULT_PROMPT,
                placeholder="Describe the plant, operating point, controller objective, and constraints...",
            )
            urdf = gr.File(
                label="Upload URDF (optional)",
                file_types=[".urdf", ".xml", ".txt"],
                type="filepath",
            )
            run_btn = gr.Button("Run Design", variant="primary")

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.Tab("Dynamics Model"):
                    model_out = gr.Markdown()
                with gr.Tab("Controller Design"):
                    controller_out = gr.Markdown()
                with gr.Tab("Structured Output"):
                    raw_json = gr.JSON()

    run_btn.click(
        fn=run_design,
        inputs=[instruction, urdf],
        outputs=[raw_json, model_out, controller_out],
    )


def main():
    demo.launch()


if __name__ == "__main__":
    main()
