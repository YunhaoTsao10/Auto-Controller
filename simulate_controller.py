import json
import math
from pathlib import Path
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


CONFIG = {
    "plant": {
        "type": "pendulum",
        "m": 1.0,
        "L": 1.0,
        "g": 9.81,
        "b": 0.05,
    },
    "controller": {
        "family": "PD",
        "theta_ref": 2.0,
        "u_eq": 8.920208,
        "Kp": 15.528492,
        "Ki": 0.0,
        "Kd": 3.95,
        "derivative_on_measurement": True,
        "integral_limit": 10.0,
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


def _try_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _extract_theta_eq_from_spec(spec_block):
    if not isinstance(spec_block, dict):
        return None

    constraints = spec_block.get("constraints", [])
    for c in constraints:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).lower()
        value = c.get("value")
        if "theta_eq" in name or "theta_ref" in name or "equilibrium_angle" in name:
            v = _try_float(value)
            if v is not None:
                return v

    text = str(spec_block.get("scenario", "")) + "\n" + str(spec_block.get("raw_request", ""))
    markers = ["theta =", "theta_eq =", "around theta ="]
    low = text.lower()
    for marker in markers:
        idx = low.find(marker)
        if idx >= 0:
            tail = text[idx + len(marker):].strip()
            token = tail.split()[0].replace(",", "")
            v = _try_float(token)
            if v is not None:
                return v

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


def load_controller_from_snapshot(path):
    p = Path(path)
    if not p.exists():
        print(f"Snapshot not found at {p}. Falling back to CONFIG['controller'] defaults.")
        return None

    data = json.loads(p.read_text(encoding="utf-8"))
    ctrl = data.get("controller", {}) or {}
    spec = data.get("spec", {}) or {}
    strategy = data.get("strategy", {}) or {}
    model = data.get("model", {}) or {}

    out = {}

    family = str(ctrl.get("family", "")).strip().upper()
    if not family:
        family = str(ctrl.get("controller_type", "")).strip().upper()
    if not family:
        strategy_family = str(strategy.get("family", strategy.get("name", ""))).strip().upper()
        if strategy_family:
            family = strategy_family
    if family:
        out["family"] = family

    params = ctrl.get("parameters", []) or []
    model_params = model.get("parameters", []) or []
    notes = ctrl.get("implementation_notes", []) or []
    param_map = {}
    for item in params + model_params:
        if isinstance(item, dict):
            k = str(item.get("name", "")).strip()
            v = item.get("value")
            if k and k not in param_map:
                param_map[k] = v

    for name in ["Kp", "Ki", "Kd", "u_eq", "theta_ref", "theta_eq", "integral_limit"]:
        if name in param_map:
            val = _try_float(param_map[name])
            if val is not None:
                out[name] = val

    if "K" in param_map:
        raw_k = param_map["K"]
        if isinstance(raw_k, list):
            try:
                out["K"] = np.array(raw_k, dtype=float).reshape(-1)
            except Exception:
                pass
        else:
            s = str(raw_k).replace("[", " ").replace("]", " ").replace(",", " ")
            vals = []
            for tok in s.split():
                v = _try_float(tok)
                if v is not None:
                    vals.append(v)
            if vals:
                out["K"] = np.array(vals, dtype=float)

    physical_state_eq = ctrl.get("physical_state_eq", []) or []
    physical_input_eq = ctrl.get("physical_input_eq", []) or []
    operating_point = model.get("operating_point", {}) or {}
    op_state_values = operating_point.get("state_values", []) or []
    op_input_values = operating_point.get("input_values", []) or []

    theta_eq = _extract_named_value(physical_state_eq, ["theta", "theta_eq"])
    if theta_eq is None:
        theta_eq = _extract_named_value(op_state_values, ["theta", "theta_eq"])
    if theta_eq is None and "theta_eq" in param_map:
        theta_eq = _try_float(param_map["theta_eq"])
    if theta_eq is None:
        theta_eq = _extract_theta_eq_from_spec(spec)
    if theta_eq is not None:
        out["theta_eq"] = theta_eq
        out["theta_ref"] = theta_eq

    u_eq = _extract_named_value(physical_input_eq, ["u", "tau", "u_eq", "tau_eq"])
    if u_eq is None:
        u_eq = _extract_named_value(op_input_values, ["u", "tau", "u_eq", "tau_eq"])
    if u_eq is None:
        for cand in ["u_eq", "tau_eq"]:
            if cand in param_map:
                u_eq = _try_float(param_map[cand])
                if u_eq is not None:
                    break
    if u_eq is not None:
        out["u_eq"] = u_eq

    for note in notes:
        s = str(note)
        if "theta_eq" in s and "theta_eq" not in out:
            parts = s.replace("=", " ").replace(":", " ").replace(",", " ").split()
            for i, tok in enumerate(parts):
                if tok.lower() == "theta_eq" and i + 1 < len(parts):
                    v = _try_float(parts[i + 1])
                    if v is not None:
                        out["theta_eq"] = v
                        out["theta_ref"] = v
                        break
        if "u_eq" in s and "u_eq" not in out:
            parts = s.replace("=", " ").replace(":", " ").replace(",", " ").split()
            for i, tok in enumerate(parts):
                if tok.lower() == "u_eq" and i + 1 < len(parts):
                    v = _try_float(parts[i + 1])
                    if v is not None:
                        out["u_eq"] = v
                        break

    print(f"Loaded controller from snapshot: {p}")
    print(f"Snapshot controller fields resolved as: {out}")
    return out


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


class PIDController:
    def __init__(self, cfg):
        self.theta_ref = cfg.get("theta_ref", 0.0)
        self.u_eq = cfg.get("u_eq", 0.0)
        self.Kp = cfg.get("Kp", 0.0)
        self.Ki = cfg.get("Ki", 0.0)
        self.Kd = cfg.get("Kd", 0.0)
        self.derivative_on_measurement = cfg.get("derivative_on_measurement", True)
        self.integral_limit = cfg.get("integral_limit", 10.0)
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def control(self, x, dt):
        theta, omega = x
        e = self.theta_ref - theta
        self.integral += e * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        d_term = -omega if self.derivative_on_measurement else 0.0
        return self.u_eq + self.Kp * e + self.Ki * self.integral + self.Kd * d_term


class LQRController:
    def __init__(self, cfg):
        self.theta_ref = cfg.get("theta_ref", 0.0)
        self.u_eq = cfg.get("u_eq", 0.0)
        K = cfg.get("K", np.array([0.0, 0.0]))
        self.K = np.array(K, dtype=float).reshape(-1)

    def reset(self):
        pass

    def control(self, x, dt):
        theta, omega = x
        dx = np.array([theta - self.theta_ref, omega], dtype=float)
        return self.u_eq - float(self.K @ dx)


def rk4_step(f, x, u, dt, plant):
    k1 = f(x, u, plant)
    k2 = f(x + 0.5 * dt * k1, u, plant)
    k3 = f(x + 0.5 * dt * k2, u, plant)
    k4 = f(x + dt * k3, u, plant)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_sim(cfg):
    plant = cfg["plant"]
    sim = cfg["sim"]
    controller_cfg = dict(cfg["controller"])

    snapshot_path = cfg.get("io", {}).get("controller_json", None)
    if snapshot_path:
        snapshot_ctrl = load_controller_from_snapshot(snapshot_path)
        if snapshot_ctrl is not None:
            controller_cfg.update(snapshot_ctrl)

    family = str(controller_cfg.get("family", "PD")).upper()
    if family in {"PD", "PID", "PI", "P"}:
        ctrl = PIDController(controller_cfg)
    elif family == "LQR":
        ctrl = LQRController(controller_cfg)
    else:
        raise ValueError(f"Unsupported controller family: {family}")

    dt = sim["dt"]
    t_final = sim["t_final"]
    x = np.array(sim["x0"], dtype=float)
    n = int(t_final / dt) + 1

    ts = np.zeros(n)
    xs = np.zeros((n, 2))
    us = np.zeros(n)
    ctrl.reset()

    for k in range(n):
        t = k * dt
        u = ctrl.control(x, dt)
        ts[k] = t
        xs[k, :] = x
        us[k] = u
        x = rk4_step(pendulum_dynamics, x, u, dt, plant)

    return ts, xs, us, controller_cfg


def compute_metrics(ts, xs, controller_cfg):
    theta = xs[:, 0]
    omega = xs[:, 1]
    theta_ref = controller_cfg.get("theta_ref", 0.0)
    e = theta - theta_ref
    final_theta = theta[-1]
    final_omega = omega[-1]

    step_mag = abs(theta_ref - theta[0])
    if step_mag < 1e-9:
        overshoot = 0.0
    else:
        if theta_ref >= theta[0]:
            peak = np.max(theta)
            overshoot = max(0.0, (peak - theta_ref) / step_mag) * 100.0
        else:
            trough = np.min(theta)
            overshoot = max(0.0, (theta_ref - trough) / step_mag) * 100.0

    band = max(0.02 * max(abs(theta_ref), 1.0), 1e-3)
    settling_time = None
    for i in range(len(ts)):
        if np.all(np.abs(theta[i:] - theta_ref) <= band):
            settling_time = ts[i]
            break

    return {
        "final_theta_rad": final_theta,
        "final_omega_rad_s": final_omega,
        "final_error_rad": e[-1],
        "overshoot_percent": overshoot,
        "settling_time_s": settling_time,
    }


def plot_results(ts, xs, us, controller_cfg):
    theta = xs[:, 0]
    omega = xs[:, 1]
    theta_ref = controller_cfg.get("theta_ref", 0.0)

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


def animate_pendulum(ts, xs, controller_cfg):
    theta_ref = controller_cfg.get("theta_ref", 0.0)
    L = CONFIG["plant"]["L"]

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


if __name__ == "__main__":
    ts, xs, us, controller_cfg = run_sim(CONFIG)
    metrics = compute_metrics(ts, xs, controller_cfg)
    print("Simulation metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")
    plot_results(ts, xs, us, controller_cfg)
    animate_pendulum(ts, xs, controller_cfg)
