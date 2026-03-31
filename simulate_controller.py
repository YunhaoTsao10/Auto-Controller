import json
import math
from pathlib import Path
from typing import Any, Callable

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


CONFIG = {
    "plant": {
        "type": "pendulum",   # "pendulum" | "mass_spring_damper"
        "m": 1.0,
        "L": 1.0,
        "g": 9.81,
        "b": 0.05,
        # MSD params
        "c": 0.4,
        "k": 4.0,
    },
    "controller": {
        "family": "PD",       # "P" | "PI" | "PD" | "PID" | "LQR"
        # Pendulum-style defaults
        "theta_ref": 2.0,
        "u_eq": 8.920208,
        "Kp": 15.528492,
        "Ki": 0.0,
        "Kd": 3.95,
        "derivative_on_measurement": True,
        "integral_limit": 10.0,
        # Generic binding
        "y_ref": None,
        "y_index": 0,
        "ydot_index": 1,
        "state_eq": None,   # for LQR if available
    },
    "sim": {
        "t_final": 20.0,
        "dt": 0.002,
        "x0": [2.2, 0.0],
    },
    "io": {
        "controller_json": "controller_snapshot.json",
    },
}


# =============================================================================
# Small helpers
# =============================================================================

def _try_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _extract_named_value(values, name_candidates):
    if not isinstance(values, list):
        return None
    lowered = {n.lower() for n in name_candidates}
    for item in values:
        if not isinstance(item, dict):
            continue
        nm = str(item.get("name", "")).lower()
        if nm in lowered:
            v = _try_float(item.get("value"))
            if v is not None:
                return v
    return None


def _named_values_to_map(values):
    out = {}
    if not isinstance(values, list):
        return out
    for item in values:
        if not isinstance(item, dict):
            continue
        nm = str(item.get("name", "")).strip()
        v = _try_float(item.get("value"))
        if nm and v is not None:
            out[nm] = v
    return out


def _parse_array_like(raw):
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            return np.array(raw, dtype=float).reshape(-1)
        except Exception:
            return None
    s = str(raw).replace("[", " ").replace("]", " ").replace(",", " ")
    vals = []
    for tok in s.split():
        v = _try_float(tok)
        if v is not None:
            vals.append(v)
    if vals:
        return np.array(vals, dtype=float)
    return None


# =============================================================================
# Snapshot loading
# =============================================================================

def load_snapshot(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        print(f"Snapshot not found at {p}. Falling back to CONFIG defaults.")
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    print(f"Loaded snapshot: {p}")
    return data


def load_plant_from_snapshot(snapshot: dict[str, Any], fallback_plant: dict[str, Any]) -> dict[str, Any]:
    plant = dict(fallback_plant)
    if snapshot is None:
        return plant

    model = snapshot.get("model", {}) or {}
    model_name = str(model.get("model_name", "")).lower()

    if "pendulum" in model_name:
        plant["type"] = "pendulum"
    elif "mass-spring-damper" in model_name or "mass spring damper" in model_name:
        plant["type"] = "mass_spring_damper"

    params = model.get("parameters", []) or []
    param_map = {}
    for item in params:
        if isinstance(item, dict):
            k = str(item.get("name", "")).strip()
            v = _try_float(item.get("value"))
            if k and v is not None:
                param_map[k] = v

    if plant["type"] == "pendulum":
        for name in ["m", "L", "g", "b"]:
            if name in param_map:
                plant[name] = param_map[name]

    elif plant["type"] == "mass_spring_damper":
        for name in ["m", "c", "k"]:
            if name in param_map:
                plant[name] = param_map[name]

    return plant


def load_controller_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    if snapshot is None:
        return None

    ctrl = snapshot.get("controller", {}) or {}
    strategy = snapshot.get("strategy", {}) or {}
    model = snapshot.get("model", {}) or {}

    out = {}

    family = str(ctrl.get("family", "")).strip().upper()
    if not family:
        family = str(ctrl.get("controller_type", "")).strip().upper()
    if not family:
        family = str(strategy.get("family", "")).strip().upper()
    if family:
        out["family"] = family

    params = ctrl.get("parameters", []) or []
    model_params = model.get("parameters", []) or []
    param_map = {}
    for item in params + model_params:
        if isinstance(item, dict):
            k = str(item.get("name", "")).strip()
            v = item.get("value")
            if k and k not in param_map:
                param_map[k] = v

    for name in ["Kp", "Ki", "Kd", "u_eq", "integral_limit"]:
        if name in param_map:
            val = _try_float(param_map[name])
            if val is not None:
                out[name] = val

    if "K" in param_map:
        K = _parse_array_like(param_map["K"])
        if K is not None:
            out["K"] = K

    operating_point = model.get("operating_point", {}) or {}
    op_state_values = operating_point.get("state_values", []) or []
    op_input_values = operating_point.get("input_values", []) or []

    # Source of truth for equilibrium input
    u_eq = _extract_named_value(op_input_values, ["u", "tau", "force", "u_eq", "tau_eq"])
    if u_eq is None:
        for cand in ["u_eq", "tau_eq"]:
            if cand in param_map:
                u_eq = _try_float(param_map[cand])
                if u_eq is not None:
                    break
    if u_eq is not None:
        out["u_eq"] = u_eq

    # Try to infer benchmark-specific reference and generic y_ref
    state_map = _named_values_to_map(op_state_values)
    if "theta" in state_map:
        out["theta_ref"] = state_map["theta"]
        out["y_ref"] = state_map["theta"]
        out["y_index"] = 0
        out["ydot_index"] = 1
        out["state_eq"] = np.array([state_map.get("theta", 0.0), state_map.get("omega", 0.0)], dtype=float)

    if "x" in state_map:
        out["x_ref"] = state_map["x"]
        out["y_ref"] = state_map["x"]
        out["y_index"] = 0
        out["ydot_index"] = 1
        out["state_eq"] = np.array([state_map.get("x", 0.0), state_map.get("v", 0.0)], dtype=float)

    # If LQR design stored physical eq explicitly, prefer that
    physical_state_eq = ctrl.get("physical_state_eq", []) or []
    if physical_state_eq:
        vals = []
        for item in physical_state_eq:
            if isinstance(item, dict):
                v = _try_float(item.get("value"))
                if v is not None:
                    vals.append(v)
        if vals:
            out["state_eq"] = np.array(vals, dtype=float)

    print(f"Snapshot controller fields resolved as: {out}")
    return out


# =============================================================================
# Benchmark-specific dynamics
# =============================================================================

def pendulum_dynamics(x, u, plant):
    theta, omega = x
    m = plant["m"]
    L = plant["L"]
    g = plant["g"]
    b = plant["b"]

    J = m * L * L
    dtheta = omega
    domega = -(g / L) * math.sin(theta) - (b / J) * omega + u / J
    return np.array([dtheta, domega], dtype=float)


def msd_dynamics(x, u, plant):
    pos, vel = x
    m = plant["m"]
    c = plant["c"]
    k = plant["k"]

    dpos = vel
    dvel = (u - c * vel - k * pos) / m
    return np.array([dpos, dvel], dtype=float)


PLANT_REGISTRY: dict[str, Callable[[np.ndarray, float, dict[str, Any]], np.ndarray]] = {
    "pendulum": pendulum_dynamics,
    "mass_spring_damper": msd_dynamics,
    # "double_pendulum": ...
}


# =============================================================================
# Controllers
# =============================================================================

class PIDController:
    """
    Benchmark-friendly local PID-family controller for 2-state second-order systems.
    Uses:
        u = u_eq + Kp*(y_ref - y) + Ki*integral(e) + Kd*d_term
    with derivative-on-measurement => d_term = - ydot
    """
    def __init__(self, cfg):
        self.y_ref = cfg.get("y_ref", cfg.get("theta_ref", cfg.get("x_ref", 0.0)))
        self.u_eq = cfg.get("u_eq", 0.0)
        self.Kp = cfg.get("Kp", 0.0)
        self.Ki = cfg.get("Ki", 0.0)
        self.Kd = cfg.get("Kd", 0.0)
        self.derivative_on_measurement = cfg.get("derivative_on_measurement", True)
        self.integral_limit = cfg.get("integral_limit", 10.0)
        self.integral = 0.0

        self.y_index = int(cfg.get("y_index", 0))
        self.ydot_index = int(cfg.get("ydot_index", 1))

    def reset(self):
        self.integral = 0.0

    def control(self, x, dt):
        y = x[self.y_index]
        ydot = x[self.ydot_index]

        e = self.y_ref - y
        self.integral += e * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        d_term = -ydot if self.derivative_on_measurement else 0.0
        return self.u_eq + self.Kp * e + self.Ki * self.integral + self.Kd * d_term


class LQRController:
    """
    Local LQR controller:
        u = u_eq - K (x - x_eq)
    """
    def __init__(self, cfg):
        self.u_eq = cfg.get("u_eq", 0.0)
        K = cfg.get("K", np.array([0.0, 0.0]))
        self.K = np.array(K, dtype=float).reshape(-1)

        state_eq = cfg.get("state_eq", None)
        if state_eq is None:
            # fallback for benchmark defaults
            ref = cfg.get("y_ref", cfg.get("theta_ref", cfg.get("x_ref", 0.0)))
            state_eq = np.array([ref, 0.0], dtype=float)
        self.state_eq = np.array(state_eq, dtype=float).reshape(-1)

    def reset(self):
        pass

    def control(self, x, dt):
        dx = np.array(x, dtype=float) - self.state_eq
        return self.u_eq - float(self.K @ dx)


# =============================================================================
# Integration
# =============================================================================

def rk4_step(f, x, u, dt, plant):
    k1 = f(x, u, plant)
    k2 = f(x + 0.5 * dt * k1, u, plant)
    k3 = f(x + 0.5 * dt * k2, u, plant)
    k4 = f(x + dt * k3, u, plant)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# =============================================================================
# Metrics
# =============================================================================

def compute_second_order_tracking_metrics(ts, xs, controller_cfg, y_label="y"):
    y = xs[:, 0]
    ydot = xs[:, 1]
    y_ref = controller_cfg.get("y_ref", controller_cfg.get("theta_ref", controller_cfg.get("x_ref", 0.0)))
    e = y - y_ref

    final_y = y[-1]
    final_ydot = ydot[-1]

    step_mag = abs(y_ref - y[0])
    if step_mag < 1e-9:
        overshoot = 0.0
    else:
        if y_ref >= y[0]:
            peak = np.max(y)
            overshoot = max(0.0, (peak - y_ref) / step_mag) * 100.0
        else:
            trough = np.min(y)
            overshoot = max(0.0, (y_ref - trough) / step_mag) * 100.0

    band = max(0.02 * max(abs(y_ref), 1.0), 1e-3)
    settling_time = None
    for i in range(len(ts)):
        if np.all(np.abs(y[i:] - y_ref) <= band):
            settling_time = ts[i]
            break

    return {
        f"final_{y_label}": final_y,
        f"final_{y_label}_dot": final_ydot,
        f"final_{y_label}_error": e[-1],
        "overshoot_percent": overshoot,
        "settling_time_s": settling_time,
    }


METRIC_REGISTRY = {
    "pendulum": lambda ts, xs, cfg: compute_second_order_tracking_metrics(ts, xs, cfg, y_label="theta_rad"),
    "mass_spring_damper": lambda ts, xs, cfg: compute_second_order_tracking_metrics(ts, xs, cfg, y_label="x_m"),
}


# =============================================================================
# Plotting / visualization
# =============================================================================

def plot_pendulum_results(ts, xs, us, controller_cfg):
    theta = xs[:, 0]
    omega = xs[:, 1]
    theta_ref = controller_cfg.get("theta_ref", controller_cfg.get("y_ref", 0.0))

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(ts, theta, label="theta")
    axs[0].axhline(theta_ref, linestyle="--", label="theta_ref")
    axs[0].set_ylabel("theta [rad]")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(ts, omega, label="omega")
    axs[1].set_ylabel("omega [rad/s]")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(ts, us, label="u")
    axs[2].set_ylabel("u [N*m]")
    axs[2].set_xlabel("time [s]")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_msd_results(ts, xs, us, controller_cfg):
    pos = xs[:, 0]
    vel = xs[:, 1]
    x_ref = controller_cfg.get("x_ref", controller_cfg.get("y_ref", 0.0))

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(ts, pos, label="x")
    axs[0].axhline(x_ref, linestyle="--", label="x_ref")
    axs[0].set_ylabel("x [m]")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(ts, vel, label="v")
    axs[1].set_ylabel("v [m/s]")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(ts, us, label="u")
    axs[2].set_ylabel("u [N]")
    axs[2].set_xlabel("time [s]")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def animate_pendulum(ts, xs, controller_cfg, plant):
    theta_ref = controller_cfg.get("theta_ref", controller_cfg.get("y_ref", 0.0))
    L = plant["L"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.2 * L, 1.2 * L)
    ax.set_ylim(-1.2 * L, 1.2 * L)
    ax.set_aspect("equal")
    ax.grid(True)

    line, = ax.plot([], [], 'o-', lw=3)
    ref_line, = ax.plot([], [], '--', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        ref_line.set_data([0, L * np.sin(theta_ref)], [0, -L * np.cos(theta_ref)])
        time_text.set_text('')
        return line, ref_line, time_text

    def update(frame):
        theta = xs[frame, 0]
        x = L * np.sin(theta)
        y = -L * np.cos(theta)
        line.set_data([0, x], [0, y])
        time_text.set_text(f't = {ts[frame]:.2f}s')
        return line, ref_line, time_text

    step = max(1, len(ts) // 600)
    ani = FuncAnimation(fig, update, frames=range(0, len(ts), step), init_func=init, blit=True, interval=20)
    plt.show()
    return ani


def animate_msd(ts, xs, controller_cfg, plant):
    pos_ref = controller_cfg.get("x_ref", controller_cfg.get("y_ref", 0.0))
    pos = xs[:, 0]

    mass_w = 0.28
    mass_h = 0.18
    wall_x = 0.0

    # Make sure the whole motion and mass body are visible
    x_min = min(wall_x - 0.1, np.min(pos) - 0.6)
    x_max = max(np.max(pos) + 0.6, pos_ref + 0.6)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True)
    ax.set_yticks([])
    ax.set_xlabel("position [m]")
    ax.set_title("Mass-Spring-Damper Animation")

    ground_y = -0.09
    spring_y = 0.10
    damper_y = -0.02

    # fixed environment
    ax.plot([x_min, x_max], [ground_y, ground_y], 'k-', lw=2)   # ground
    ax.plot([wall_x, wall_x], [-0.40, 0.40], 'k-', lw=5)        # wall

    # reference marker
    ref_line = ax.axvline(pos_ref, linestyle='--', lw=1.5, label="reference")

    # animated artists
    spring_line, = ax.plot([], [], lw=2, label="spring")
    damper_line1, = ax.plot([], [], lw=2, color="tab:orange")
    damper_line2, = ax.plot([], [], lw=2, color="tab:green")
    damper_line3, = ax.plot([], [], lw=2, color="tab:red")

    mass_patch = plt.Rectangle((0, 0), mass_w, mass_h, fill=True, lw=2, edgecolor="black", facecolor="black")
    ax.add_patch(mass_patch)

    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    pos_text = ax.text(0.02, 0.82, '', transform=ax.transAxes)

    def make_spring(x0, x1, y, coils=8, amp=0.06):
        # If too short, just draw a straight segment
        if x1 <= x0 + 1e-4:
            return np.array([x0, x1]), np.array([y, y])

        xs_spring = np.linspace(x0, x1, 2 * coils + 1)
        ys_spring = np.full_like(xs_spring, y)
        for i in range(1, len(xs_spring) - 1):
            ys_spring[i] += amp if i % 2 else -amp
        ys_spring[0] = y
        ys_spring[-1] = y
        return xs_spring, ys_spring

    def init():
        spring_line.set_data([], [])
        damper_line1.set_data([], [])
        damper_line2.set_data([], [])
        damper_line3.set_data([], [])
        mass_patch.set_xy((0.8 - mass_w / 2, -mass_h / 2))
        time_text.set_text('')
        pos_text.set_text('')
        return (
            spring_line, damper_line1, damper_line2, damper_line3,
            mass_patch, ref_line, time_text, pos_text
        )

    def update(frame):
        x_center = xs[frame, 0]
        mass_left = x_center - mass_w / 2

        # update mass
        mass_patch.set_xy((mass_left, -mass_h / 2))

        # spring: wall -> mass left edge
        sx0 = wall_x
        sx1 = mass_left
        sx, sy = make_spring(sx0, sx1, spring_y)
        spring_line.set_data(sx, sy)

        # damper: wall -> mass left edge
        # split into rod + body + rod
        total_len = max(mass_left - wall_x, 1e-4)
        body_left = wall_x + 0.38 * total_len
        body_right = wall_x + 0.68 * total_len

        # rod from wall to body
        damper_line1.set_data([wall_x, body_left], [damper_y, damper_y])

        # body
        damper_line2.set_data(
            [body_left, body_left, body_right, body_right],
            [damper_y - 0.06, damper_y + 0.06, damper_y + 0.06, damper_y - 0.06]
        )

        # rod from body to mass
        damper_line3.set_data([body_right, mass_left], [damper_y, damper_y])

        time_text.set_text(f't = {ts[frame]:.2f}s')
        pos_text.set_text(f'x = {x_center:.3f} m')
        return (
            spring_line, damper_line1, damper_line2, damper_line3,
            mass_patch, ref_line, time_text, pos_text
        )

    step = max(1, len(ts) // 600)
    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, len(ts), step),
        init_func=init,
        blit=True,
        interval=20,
    )
    ax.legend(loc="upper right")
    plt.show()
    return ani


PLOT_REGISTRY = {
    "pendulum": plot_pendulum_results,
    "mass_spring_damper": plot_msd_results,
}

ANIMATION_REGISTRY = {
    "pendulum": animate_pendulum,
    "mass_spring_damper": animate_msd,
}


# =============================================================================
# Main simulation loop
# =============================================================================

def run_sim(cfg):
    plant = dict(cfg["plant"])
    sim = cfg["sim"]
    controller_cfg = dict(cfg["controller"])

    snapshot_path = cfg.get("io", {}).get("controller_json", None)
    snapshot = load_snapshot(snapshot_path) if snapshot_path else None

    if snapshot is not None:
        plant = load_plant_from_snapshot(snapshot, plant)
        snapshot_ctrl = load_controller_from_snapshot(snapshot)
        if snapshot_ctrl is not None:
            controller_cfg.update(snapshot_ctrl)

    plant_type = plant.get("type", "pendulum")
    if plant_type not in PLANT_REGISTRY:
        raise ValueError(f"Unsupported plant type: {plant_type}")

    family = str(controller_cfg.get("family", "PD")).upper()
    if family in {"PD", "PID", "PI", "P"}:
        ctrl = PIDController(controller_cfg)
    elif family == "LQR":
        ctrl = LQRController(controller_cfg)
    else:
        raise ValueError(f"Unsupported controller family: {family}")

    dynamics_fn = PLANT_REGISTRY[plant_type]

    dt = sim["dt"]
    t_final = sim["t_final"]
    x = np.array(sim["x0"], dtype=float)
    n = int(t_final / dt) + 1

    ts = np.zeros(n)
    xs = np.zeros((n, len(x)))
    us = np.zeros(n)
    ctrl.reset()

    for k in range(n):
        t = k * dt
        u = ctrl.control(x, dt)
        ts[k] = t
        xs[k, :] = x
        us[k] = u
        x = rk4_step(dynamics_fn, x, u, dt, plant)

    return ts, xs, us, controller_cfg, plant


def compute_metrics(ts, xs, controller_cfg, plant):
    plant_type = plant.get("type", "pendulum")
    if plant_type not in METRIC_REGISTRY:
        raise ValueError(f"No metric function for plant type: {plant_type}")
    return METRIC_REGISTRY[plant_type](ts, xs, controller_cfg)


def plot_results(ts, xs, us, controller_cfg, plant):
    plant_type = plant.get("type", "pendulum")
    if plant_type not in PLOT_REGISTRY:
        raise ValueError(f"No plotting function for plant type: {plant_type}")
    return PLOT_REGISTRY[plant_type](ts, xs, us, controller_cfg)


def animate_results(ts, xs, controller_cfg, plant):
    plant_type = plant.get("type", "pendulum")
    if plant_type not in ANIMATION_REGISTRY:
        print(f"No animation registered for plant type: {plant_type}")
        return None
    return ANIMATION_REGISTRY[plant_type](ts, xs, controller_cfg, plant)


if __name__ == "__main__":
    ts, xs, us, controller_cfg, plant = run_sim(CONFIG)
    metrics = compute_metrics(ts, xs, controller_cfg, plant)
    print("Simulation metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")
    plot_results(ts, xs, us, controller_cfg, plant)
    animate_results(ts, xs, controller_cfg, plant)