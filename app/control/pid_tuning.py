import re
import numpy as np

from app.schemas import PIDDesignSpec, ControllerDesign, Parameter, DynamicsModel


# ---------------------------
# Basic target conversion
# ---------------------------

def damping_ratio_from_overshoot(mp_pct: float) -> float:
    """Percent overshoot -> damping ratio zeta (for 0<zeta<1)."""
    mp = max(1e-6, min(0.99, float(mp_pct) / 100.0))
    ln_mp = np.log(mp)
    zeta = -ln_mp / np.sqrt(np.pi**2 + ln_mp**2)
    return float(np.clip(zeta, 0.1, 0.95))


def second_order_targets(settle_time_s: float, overshoot_pct: float) -> tuple[float, float]:
    """Use 2% settling-time approximation: Ts ≈ 4 / (zeta * wn)."""
    Ts = max(0.05, float(settle_time_s))
    zeta = damping_ratio_from_overshoot(float(overshoot_pct))
    wn = 4.0 / (zeta * Ts)
    return zeta, wn


# ---------------------------
# Small parsing helpers
# ---------------------------

def _extract_float_from_parameters(model: DynamicsModel, name: str) -> float | None:
    for p in model.parameters:
        if p.name.strip().lower() == name.lower():
            try:
                return float(p.value)
            except Exception:
                return None
    return None


def _search_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text.replace("−", "-"), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_operating_point_theta(model: DynamicsModel) -> float | None:
    """
    Try to infer a pendulum operating point theta_eq [rad] from assumptions/state-space notes.
    Examples accepted:
      - "linearized around theta = 1.57 rad"
      - "Linearization around theta_eq = 0.0"
      - "equilibrium at theta*=..."
    """
    op_lines = []
    if getattr(model, 'operating_point', None) is not None:
        op_lines.extend([f"{v.name}={v.value}" for v in model.operating_point.state_values])
        op_lines.extend([f"{v.name}={v.value}" for v in model.operating_point.input_values])
    llm_lines = []
    if getattr(model, 'local_linear_model', None) is not None:
        llm = model.local_linear_model
        llm_lines.extend([llm.valid_near, llm.state_definition, llm.input_definition])
        llm_lines.extend(llm.notes or [])
    text = "\n".join((model.assumptions or []) + (model.state_space or []) + (model.ode or []) + op_lines + llm_lines)
    patterns = [
        r"theta(?:_eq|\*)?\s*=\s*([+-]?\d*\.?\d+)",
        r"around\s+theta\s*=\s*([+-]?\d*\.?\d+)",
        r"linearization\s+point[^\n]*?([+-]?\d*\.?\d+)\s*rad",
        r"equilibrium[^\n]*?theta[^\n]*?([+-]?\d*\.?\d+)",
    ]
    for pat in patterns:
        val = _search_float(pat, text)
        if val is not None:
            return val
    return None


def _infer_pendulum_local_coeffs(model: DynamicsModel) -> dict | None:
    """
    For a simple pendulum model
        theta_dot = omega
        omega_dot = -(g/L) sin(theta) - (b/J) omega + (1/J) u,
    infer the local deviation model around theta_eq:
        d2(delta_theta)/dt2 + a1 d(delta_theta)/dt + a0 delta_theta = b0 delta_u
    with
        a1 = b / J, a0 = (g/L) cos(theta_eq), b0 = 1 / J, J = m L^2,
        u_eq = m g L sin(theta_eq)
    """
    structured_dyn = [f"{eq.lhs} = {eq.rhs}" for eq in getattr(model, 'dynamics', [])]
    model_blob = " ".join(
        [model.model_name.lower()] + [line.lower() for line in (model.ode or [])] + [line.lower() for line in structured_dyn] + [line.lower() for line in (model.assumptions or [])]
    )
    looks_like_pendulum = (
        "pendulum" in model_blob
        or ("sin(theta)" in model_blob and any(v.name.lower() == "theta" for v in model.states))
    )
    if not looks_like_pendulum:
        return None

    m = _extract_float_from_parameters(model, "m")
    L = _extract_float_from_parameters(model, "L")
    g = _extract_float_from_parameters(model, "g")
    b = _extract_float_from_parameters(model, "b")
    theta_eq = _extract_operating_point_theta(model)

    if None in {m, L, g, b}:
        return None
    if theta_eq is None:
        # fall back to origin if the model looks like the standard small-angle pendulum design case
        theta_eq = 0.0

    J = float(m) * float(L) ** 2
    if J <= 0:
        return None

    a1 = float(b) / J
    a0 = (float(g) / float(L)) * float(np.cos(theta_eq))
    b0 = 1.0 / J
    u_eq = float(m) * float(g) * float(L) * float(np.sin(theta_eq))

    return {
        "a1": a1,
        "a0": a0,
        "b0": b0,
        "theta_eq": float(theta_eq),
        "u_eq": u_eq,
        "J": J,
    }


# ---------------------------
# Canonical second-order form extraction
# ---------------------------

def extract_second_order_siso_form(model: DynamicsModel) -> tuple[float, float, float] | None:
    """
    Try to extract canonical coefficients (a1, a0, b0) for:
        y'' + a1 y' + a0 y = b0 u

    Preferred source:
      0) infer from pendulum operating point if recognizable
      1) parameters named a1, a0, b0
      2) simple parse from state_space/ode text lines
    """
    inferred = _infer_pendulum_local_coeffs(model)
    if inferred is not None:
        return inferred["a1"], inferred["a0"], inferred["b0"]

    a1 = _extract_float_from_parameters(model, "a1")
    a0 = _extract_float_from_parameters(model, "a0")
    b0 = _extract_float_from_parameters(model, "b0")
    if a1 is not None and a0 is not None and b0 is not None:
        return a1, a0, b0

    text_blob = "\n".join((model.state_space or []) + (model.ode or []))
    m = re.search(
        r"G\s*\(\s*s\s*\)\s*=\s*([+-]?\d*\.?\d+)\s*/\s*\(\s*s\^?2\s*([+-]\s*\d*\.?\d+)\s*s\s*([+-]\s*\d*\.?\d+)\s*\)",
        text_blob.replace("−", "-"),
        flags=re.IGNORECASE,
    )
    if m:
        try:
            b0 = float(m.group(1).replace(" ", ""))
            a1 = float(m.group(2).replace(" ", ""))
            a0 = float(m.group(3).replace(" ", ""))
            return a1, a0, b0
        except Exception:
            pass

    return None


# ---------------------------
# Packaging result
# ---------------------------

def build_pid_controller_design(
    pid_spec: PIDDesignSpec,
    Kp: float,
    Ki: float,
    Kd: float,
    extra_assumptions: list[str] | None = None,
    tuning_notes: list[str] | None = None,
    bias_input: float | None = None,
) -> ControllerDesign:
    u = pid_spec.control_input
    y = pid_spec.target_variable
    r = pid_spec.reference_name

    structure = pid_spec.structure.upper().strip()
    if structure not in {"P", "PI", "PD", "PID"}:
        structure = "PID"

    law = [f"e = {r} - {y}"]
    bias_str = f"{bias_input:.6f} + " if bias_input is not None and abs(bias_input) > 1e-12 else ""

    if structure == "P":
        law.append(f"{u} = {bias_str}{Kp:.6f}*e")
    elif structure == "PI":
        law.append(f"{u} = {bias_str}{Kp:.6f}*e + {Ki:.6f}*integral(e)")
    elif structure == "PD":
        if pid_spec.use_derivative_on_measurement:
            law.append(f"{u} = {bias_str}{Kp:.6f}*e - {Kd:.6f}*d({y})/dt")
        else:
            law.append(f"{u} = {bias_str}{Kp:.6f}*e + {Kd:.6f}*d(e)/dt")
    else:
        if pid_spec.use_derivative_on_measurement:
            law.append(f"{u} = {bias_str}{Kp:.6f}*e + {Ki:.6f}*integral(e) - {Kd:.6f}*d({y})/dt")
        else:
            law.append(f"{u} = {bias_str}{Kp:.6f}*e + {Ki:.6f}*integral(e) + {Kd:.6f}*d(e)/dt")

    params = [
        Parameter(name="Kp", meaning="proportional gain", unit="controller-dependent", value=f"{Kp:.6f}", notes="model-based tuning"),
        Parameter(name="Ki", meaning="integral gain", unit="controller-dependent", value=f"{Ki:.6f}", notes="0 for P/PD; heuristic initial value for PI/PID"),
        Parameter(name="Kd", meaning="derivative gain", unit="controller-dependent", value=f"{Kd:.6f}", notes="0 for P/PI"),
    ]
    if bias_input is not None and abs(bias_input) > 1e-12:
        params.append(
            Parameter(
                name="u_eq",
                meaning="equilibrium/bias input required to hold the operating point",
                unit="same as control input",
                value=f"{bias_input:.6f}",
                notes="Add this feedforward term before the feedback correction.",
            )
        )

    notes = [
        f"Controller form: {pid_spec.form}",
        f"Selected structure: {structure}",
        f"dt = {pid_spec.dt_s:.4f} s",
        f"Anti-windup: {pid_spec.anti_windup}",
        "Derivative filter is intentionally omitted in this simplified version.",
    ]
    if pid_spec.actuator_limit is not None:
        notes.append(f"Apply saturation: |{u}| <= {pid_spec.actuator_limit:.6f}")
    if tuning_notes:
        notes.extend(tuning_notes)

    assumptions = [
        "Plant is approximated as a SISO second-order linear model near an operating point.",
        "Canonical form used: G(s) = b0 / (s^2 + a1 s + a0).",
    ]
    if extra_assumptions:
        assumptions.extend(extra_assumptions)

    required_signals = [pid_spec.reference_name, pid_spec.target_variable]
    if structure in {"PD", "PID"}:
        if pid_spec.use_derivative_on_measurement:
            required_signals.append(f"d({pid_spec.target_variable})/dt")
        else:
            required_signals.append("d(error)/dt")

    return ControllerDesign(
        controller_type=structure,
        required_signals=required_signals,
        control_law=law,
        parameters=params,
        assumptions=assumptions,
        implementation_notes=notes,
    )


# ---------------------------
# Main solver (generic 2nd-order SISO)
# ---------------------------

def solve_pid_from_model(model: DynamicsModel, pid_spec: PIDDesignSpec) -> ControllerDesign:
    structure = (pid_spec.structure or "PID").upper().strip()
    if structure not in {"P", "PI", "PD", "PID"}:
        structure = "PID"
    pid_spec.structure = structure

    zeta, wn = second_order_targets(pid_spec.settle_time_s, pid_spec.overshoot_pct)

    coeffs = extract_second_order_siso_form(model)
    if coeffs is None:
        raise ValueError(
            "Could not extract canonical second-order SISO form. "
            "Please provide model parameters a1, a0, b0 (recommended), "
            "or include a parseable transfer function line like G(s)=b0/(s^2+a1s+a0)."
        )

    a1, a0, b0 = coeffs
    if abs(b0) < 1e-9:
        raise ValueError("Input gain b0 is too close to zero; cannot tune controller.")

    pendulum_info = _infer_pendulum_local_coeffs(model)
    bias_input = pendulum_info["u_eq"] if pendulum_info is not None else None

    Kd = (2.0 * zeta * wn - a1) / b0
    Kp = (wn**2 - a0) / b0
    Ki = 0.0

    tuning_notes = [
        f"Target mapping (2% settling-time): zeta={zeta:.3f}, wn={wn:.3f} rad/s",
        f"Canonical coefficients used: a1={a1:.6f}, a0={a0:.6f}, b0={b0:.6f}",
        "PD gains from exact second-order coefficient matching on the local deviation model.",
    ]
    extra_assumptions = [
        "This tuning is intended as an initial design and should be validated in simulation.",
    ]

    if pendulum_info is not None:
        tuning_notes.append(
            f"Pendulum operating point detected: theta_eq={pendulum_info['theta_eq']:.6f} rad, u_eq={pendulum_info['u_eq']:.6f}."
        )
        extra_assumptions.append(
            "Controller is designed for the deviation dynamics around the detected operating point; implement u = u_eq + feedback."
        )

    if structure in {"PI", "PID"}:
        Ki = 0.1 * max(0.1, wn) * max(0.0, Kp)
        tuning_notes.append("Ki is a conservative heuristic initial value (not exact pole matching).")

    if structure == "P":
        Ki = 0.0
        Kd = 0.0
    elif structure == "PI":
        Kd = 0.0
    elif structure == "PD":
        Ki = 0.0

    if Kp < 0:
        raise ValueError(
            f"Computed Kp={Kp:.6f} < 0 from the inferred local model. "
            "This usually means the operating point/model sign convention is inconsistent with the requested target dynamics."
        )
    if structure in {"PD", "PID"} and Kd < 0:
        tuning_notes.append("Kd < 0 from target mapping; clamped to 0.0 (target is too aggressive or model damping is already high).")
        Kd = 0.0
    if structure in {"PI", "PID"} and Ki < 0:
        tuning_notes.append("Ki < 0 after heuristic; clamped to 0.0.")
        Ki = 0.0

    return build_pid_controller_design(
        pid_spec=pid_spec,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        extra_assumptions=extra_assumptions,
        tuning_notes=tuning_notes,
        bias_input=bias_input,
    )
