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

    A: Matrix
    B: Matrix
    C: Matrix | None = None
    D: Matrix | None = None

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

    # Backward-compatible legacy fields
    ode: list[str] = Field(default_factory=list, description="plain-text ODE lines")
    state_space: list[str] = Field(default_factory=list, description="optional linearized form notes")
    notes: list[str] = Field(default_factory=list)
