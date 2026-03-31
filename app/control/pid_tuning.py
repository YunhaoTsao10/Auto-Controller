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
# Canonical second-order form extraction
# ---------------------------

def extract_equilibrium_bias_input(model: DynamicsModel) -> float | None:
    op = getattr(model, "operating_point", None)
    if op is None:
        return None

    if not op.input_values:
        return None

    # for current SISO scope, use the first equilibrium input
    try:
        return float(op.input_values[0].value)
    except Exception:
        return None

def extract_second_order_siso_form(model: DynamicsModel) -> tuple[float, float, float] | None:
    cr = getattr(model, "controller_ready", None)
    if cr is None:
        return None
    if cr.model_class != "second_order_siso":
        return None
    if cr.second_order is None:
        return None
    if not cr.second_order.valid:
        return None

    so = cr.second_order
    return so.a1, so.a0, so.b0


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
    bias_input=extract_equilibrium_bias_input(model)

    coeffs = extract_second_order_siso_form(model)
    if coeffs is None:
        raise ValueError(
            "Modeling output is missing a valid controller_ready.second_order block for second-order SISO PID synthesis."
        )

    a1, a0, b0 = coeffs
    if abs(b0) < 1e-9:
        raise ValueError("Input gain b0 is too close to zero; cannot tune controller.")

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
            f"Computed Kp={Kp:.6f} < 0 from the controller-ready local model. "
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
