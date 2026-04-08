from pydantic import BaseModel, Field

from app.schemas.modeling import Parameter, Matrix, NamedValue


class ControllerDesign(BaseModel):
    controller_type: str = Field(..., description="e.g., LQR, PID, MPC")
    controller_name: str | None = None
    required_signals: list[str] = Field(default_factory=list, description="signals needed online")
    control_law: list[str] = Field(default_factory=list, description="equations for u, plain text")
    parameters: list[Parameter] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)

    designed_for_operating_point: str | None = None
    designed_for_linear_model: str | None = None
    uses_deviation_variables: bool | None = None
    state_order: list[str] = Field(default_factory=list)
    input_order: list[str] = Field(default_factory=list)
    physical_state_eq: list[NamedValue] = Field(default_factory=list)
    physical_input_eq: list[NamedValue] = Field(default_factory=list)


class LQRSpec(BaseModel):
    discrete: bool = False
    dt_s: float | None = None

    operating_point_name: str
    state_order: list[str]
    input_order: list[str] = Field(default_factory=list)
    input_name: str | None = None

    uses_deviation_variables: bool = True
    deviation_state_definition: str = "delta_x = x - x_eq"
    deviation_input_definition: str = "delta_u = u - u_eq"
    final_control_law_template: str = "u_phys = u_eq + delta_u"

    physical_state_eq: list[NamedValue] = Field(default_factory=list)
    physical_input_eq: list[NamedValue] = Field(default_factory=list)

    A: Matrix
    B: Matrix
    Q: Matrix
    R: Matrix
    notes: list[str] = Field(default_factory=list)


class PIDDesignSpec(BaseModel):
    structure: str = Field(..., description="P / PI / PD / PID")
    form: str = Field(..., description="parallel / ideal")
    target_variable: str = Field(..., description="e.g. theta")
    control_input: str = Field(..., description="e.g. tau")
    reference_name: str = Field(..., description="e.g. theta_ref")

    dt_s: float = Field(0.01, description="control timestep")
    settle_time_s: float = Field(2.0, description="desired settling time")
    overshoot_pct: float = Field(10.0, description="desired percent overshoot")
    steady_state_error_target: float = Field(0.0, description="desired steady-state error")
    actuator_limit: float | None = Field(None, description="abs saturation limit")

    use_derivative_on_measurement: bool = True
    derivative_filter: bool = True
    derivative_filter_N: float = 10.0
    anti_windup: str = Field("clamping", description="none / clamping / back_calculation")

    notes: list[str] = Field(default_factory=list)
