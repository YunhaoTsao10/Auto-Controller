from pydantic import BaseModel, Field
from typing import Literal


class Variable(BaseModel):
    name: str = Field(..., description="symbol name, e.g., x, v, theta, omega")
    meaning: str = Field(..., description="physical meaning")
    unit: str = Field(..., description="SI unit, e.g., m, rad, m/s")


class Parameter(BaseModel):
    name: str = Field(..., description="parameter symbol, e.g., m, k, c, g, L")
    meaning: str
    unit: str
    value: str = Field(..., description="string value; if unknown, write 'unknown'")
    notes: str = ""


class Equation(BaseModel):
    lhs: str = Field(..., description="left-hand side, e.g. theta_dot")
    rhs: str = Field(..., description="right-hand side, e.g. omega")
    notes: str = ""


class NamedValue(BaseModel):
    name: str
    value: float
    unit: str = ""
    notes: str = ""


class Matrix(BaseModel):
    rows: int
    cols: int
    data: list[list[float]]


class OperatingPoint(BaseModel):
    name: str = Field(..., description="e.g. upright_eq, hover_trim, cruise_trim")
    kind: Literal["equilibrium", "trim", "reference", "trajectory_sample"] = "equilibrium"
    state_values: list[NamedValue] = Field(default_factory=list)
    input_values: list[NamedValue] = Field(default_factory=list)
    output_values: list[NamedValue] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    residual_description: str = ""


class LinearizationInfo(BaseModel):
    valid_near: str = Field(..., description="operating-point name or description")
    state_order: list[str] = Field(default_factory=list)
    input_order: list[str] = Field(default_factory=list)
    output_order: list[str] = Field(default_factory=list)

    state_definition: str = Field(default="delta_x = x - x_eq")
    input_definition: str = Field(default="delta_u = u - u_eq")
    output_definition: str = Field(default="delta_y = y - y_eq")

    feedback_coordinates: Literal["physical", "deviation"] = "deviation"
    final_control_law_template: str = Field(
        default="u_phys = u_eq + delta_u",
        description="How feedback-space control should be mapped back to physical input."
    )

    A: Matrix
    B: Matrix
    C: Matrix | None = None
    D: Matrix | None = None

    notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    sanity_checks: list[str] = Field(default_factory=list)


class SecondOrderCanonicalInfo(BaseModel):
    valid: bool = Field(..., description="Whether a local second-order SISO deviation-form representation is valid")
    form: Literal["delta_y_ddot + a1*delta_y_dot + a0*delta_y = b0*delta_u"] = \
        "delta_y_ddot + a1*delta_y_dot + a0*delta_y = b0*delta_u"

    y_name: str = Field(..., description="controlled output in absolute coordinates, e.g. theta or x")
    ydot_name: str = Field(..., description="time derivative of the controlled output, e.g. omega or v")

    y_index_in_state: int = Field(..., description="index in local_linear_model.state_order")
    ydot_index_in_state: int = Field(..., description="index in local_linear_model.state_order")

    a0: float = Field(..., description="coefficient multiplying delta_y")
    a1: float = Field(..., description="coefficient multiplying delta_y_dot")
    b0: float = Field(..., description="coefficient multiplying delta_u")

    assumptions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ControllerReadyModel(BaseModel):
    model_class: Literal["second_order_siso", "general_siso", "general_mimo", "unsupported"] = "unsupported"

    second_order: SecondOrderCanonicalInfo | None = None

    notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    sanity_checks: list[str] = Field(default_factory=list)


class DynamicsModel(BaseModel):
    model_name: str = Field(..., description="e.g., Spring-Mass-Damper, Simple Pendulum")
    model_type: Literal["nonlinear_ode", "linear_state_space", "transfer_function", "hybrid"] = "nonlinear_ode"

    states: list[Variable]
    inputs: list[Variable]
    outputs: list[Variable]
    parameters: list[Parameter] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    dynamics: list[Equation] = Field(default_factory=list, description="structured dynamics equations")
    output_equations: list[Equation] = Field(default_factory=list)
    operating_point: OperatingPoint | None = None
    local_linear_model: LinearizationInfo | None = None

    controller_ready: ControllerReadyModel | None = None

    # Backward-compatible legacy fields
    ode: list[str] = Field(default_factory=list, description="plain-text ODE lines")
    state_space: list[str] = Field(default_factory=list, description="optional linearized form notes")
    notes: list[str] = Field(default_factory=list)