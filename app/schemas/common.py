from pydantic import BaseModel, Field


class DesignTarget(BaseModel):
    dt_s: str = "0.01"
    settle_time_s: str = "2.0"
    overshoot_pct: str = "10"
    torque_limit_nm: str = "2.0"
    theta_range_rad: str = "0.2"


class ConstraintKV(BaseModel):
    name: str = Field(..., description="constraint key, e.g. torque_limit_nm")
    value: str = Field(..., description="constraint value (string)")
    unit: str | None = Field(None, description="unit if applicable, e.g. Nm, ms")
    kind: str = Field("hard", description="hard or soft constraint")


class Spec(BaseModel):
    scenario: str = Field(..., description="the control scenario to be executed")
    constraints: list[ConstraintKV] = Field(default_factory=list)
    urdf: str | None = None
    design_targets: DesignTarget = Field(default_factory=DesignTarget)


class Strategy(BaseModel):
    family: str = Field(..., description="e.g., PID, LQR, MPC")
    reason: str = Field(..., description="short rationale")


class StrategyOutput(BaseModel):
    spec: Spec
    strategy: Strategy


class QCReport(BaseModel):
    step: str
    passed: bool
    notes: str = ""