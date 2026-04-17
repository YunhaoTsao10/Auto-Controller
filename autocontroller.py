import time
import json
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


# =========================
# Config
# =========================
ROBOT_XML = "robot.xml"
WAYPOINT_JSON = "optimized_viewpoints.json"
SNAPSHOT_JSON = "controller_snapshot.json"

SCALE = 1.0
RUN_SPEED = 1.0
CONTROL_HZ = 500
RENDER_HZ = 60
MAX_STEPS_PER_FRAME = 64
IDLE_SLEEP = 0.0005

POS_DONE_TOL = 0.2
VEL_DONE_TOL = 0.1
STOP_TIME = 0.0

THRUSTER_NAMES = [
    "thruster_px", "thruster_nx",
    "thruster_py", "thruster_ny",
    "thruster_pz", "thruster_nz",
]
RW_NAMES = ["rw_x", "rw_y", "rw_z"]
ALL_ACT_NAMES = THRUSTER_NAMES + RW_NAMES

FIRE_THRESH = 0.1


# =========================
# Utilities
# =========================
def quat_to_R(q):
    """q=[w,x,y,z] -> R(world <- body)."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=float)
    return R


def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_mul(qa, qb):
    wa, xa, ya, za = qa
    wb, xb, yb, zb = qb
    return np.array([
        wa*wb - xa*xb - ya*yb - za*zb,
        wa*xb + xa*wb + ya*zb - za*yb,
        wa*yb - xa*zb + ya*wb + za*xb,
        wa*zb + xa*yb - ya*xb + za*wb
    ], dtype=float)


def attitude_error_vec(q_current, q_des):
    """
    Small local attitude error vector.

    q_current, q_des are both [w, x, y, z], representing body->world orientation.
    We compute q_err = q_des^{-1} * q_current, then use the small-angle approximation
    e ≈ 2 * vec(q_err), with sign chosen so q_err_w >= 0.
    """
    q_current = np.asarray(q_current, dtype=float)
    q_des = np.asarray(q_des, dtype=float)

    q_current = q_current / max(np.linalg.norm(q_current), 1e-12)
    q_des = q_des / max(np.linalg.norm(q_des), 1e-12)

    q_err = quat_mul(quat_conj(q_des), q_current)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def step_until(model, data, target_sim_time):
    steps = 0
    while data.time + 1e-12 < target_sim_time and steps < MAX_STEPS_PER_FRAME:
        mujoco.mj_step(model, data)
        steps += 1
    return steps


def reached(p, v, p_goal):
    ep = p - p_goal
    return (np.linalg.norm(ep) < POS_DONE_TOL) and (np.linalg.norm(v) < VEL_DONE_TOL)


def clamp_actuator_commands(u_cmd):
    """
    Thrusters: clamp to [0, 20]
    Reaction wheels: clamp to [-5, 5]
    """
    u = np.array(u_cmd, dtype=float).copy()
    u[:6] = np.clip(u[:6], 0.0, 20.0)
    u[6:] = np.clip(u[6:], -5.0, 5.0)
    return u


def set_axis_with_min(u, act_map, pos_name, neg_name, F_axis, Fsat=20.0, min_fire=0.0):
    """Used only when snapshot gives 6 generalized inputs instead of 9 actuator inputs."""
    if abs(F_axis) < min_fire:
        return
    pos_id = act_map[pos_name]
    neg_id = act_map[neg_name]
    if F_axis >= 0.0:
        if pos_id >= 0:
            u[pos_id] = min(F_axis, Fsat)
    else:
        if neg_id >= 0:
            u[neg_id] = min(-F_axis, Fsat)


# =========================
# Snapshot loading
# =========================
def _extract_named_values(named_list):
    out = {}
    for item in named_list or []:
        out[item["name"]] = float(item["value"])
    return out


def load_lqr_from_snapshot(snapshot_path):
    snap = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    ctrl = snap["controller"]
    model = snap["model"]

    if ctrl.get("controller_type") != "LQR":
        raise RuntimeError("controller_snapshot.json is not an LQR controller snapshot.")

    state_order = ctrl.get("state_order") or model["local_linear_model"]["state_order"]
    input_order = ctrl.get("input_order") or model["local_linear_model"]["input_order"]

    # K is in controller.parameters
    K = None
    for p in ctrl.get("parameters", []):
        if p.get("name") == "K":
            K = np.array(json.loads(p["value"]), dtype=float)
            break
    if K is None:
        raise RuntimeError("Could not find K in controller.parameters.")

    # Prefer controller.physical_state_eq / physical_input_eq if present
    xeq_named = ctrl.get("physical_state_eq")
    ueq_named = ctrl.get("physical_input_eq")

    if not xeq_named:
        # fallback: derive from controller parameters
        xeq_named = []
        state_param_map = {}
        for p in ctrl.get("parameters", []):
            nm = p.get("name", "")
            if nm.endswith("_eq"):
                state_param_map[nm[:-3]] = float(p["value"])
        for s in state_order:
            if s in state_param_map:
                xeq_named.append({"name": s, "value": state_param_map[s]})

    if not ueq_named:
        ueq_named = []
        input_param_map = {}
        for p in ctrl.get("parameters", []):
            nm = p.get("name", "")
            if nm.endswith("_eq"):
                input_param_map[nm[:-3]] = float(p["value"])
        for u in input_order:
            if u in input_param_map:
                ueq_named.append({"name": u, "value": input_param_map[u]})

    xeq_map = _extract_named_values(xeq_named)
    ueq_map = _extract_named_values(ueq_named)

    x_eq = np.array([xeq_map.get(name, 0.0) for name in state_order], dtype=float)
    u_eq = np.array([ueq_map.get(name, 0.0) for name in input_order], dtype=float)

    return {
        "state_order": state_order,
        "input_order": input_order,
        "K": K,
        "x_eq": x_eq,
        "u_eq": u_eq,
        "snapshot": snap,
    }


# =========================
# Waypoints
# =========================
def load_waypoints(path):
    file = json.loads(Path(path).read_text(encoding="utf-8"))
    WPS = file.get("waypoints", [])
    if not WPS:
        raise RuntimeError("optimized_viewpoints.json has no 'waypoints' entries")

    vp_pos = np.asarray([wp["pos"] for wp in WPS], dtype=float)
    vp_vel = np.asarray([wp.get("vel", [0.0, 0.0, 0.0]) for wp in WPS], dtype=float)
    vp_quat = np.asarray([wp.get("quat", [1.0, 0.0, 0.0, 0.0]) for wp in WPS], dtype=float)
    vp_omega = np.asarray([wp.get("omega", [0.0, 0.0, 0.0]) for wp in WPS], dtype=float)

    viewpoints_pos = [p * SCALE for p in vp_pos]
    viewpoints_vel = [v for v in vp_vel]
    viewpoints_quat = [q for q in vp_quat]
    viewpoints_omega = [w for w in vp_omega]

    return viewpoints_pos, viewpoints_vel, viewpoints_quat, viewpoints_omega


# =========================
# State construction
# =========================
def build_state_vector(state_order, pW, vW, qWB, wB, q_des, wheel_state=None):
    """
    Construct x_phys in the order expected by the snapshot.

    Supports:
    - position aliases: p_x / px / x, etc.
    - velocity aliases: v_x / vx, etc.
    - local attitude error aliases:
        e_x/e_y/e_z
        e_rx/e_ry/e_rz
        dtheta_x/dtheta_y/dtheta_z
        phi_x/phi_y/phi_z
        qx/qy/qz   (legacy local-error alias, NOT full quaternion)
    - angular velocity aliases:
        w_x/w_y/w_z
        wx/wy/wz
        omega_x/omega_y/omega_z
    - optional reaction-wheel internal states if present
    """
    att_err = attitude_error_vec(qWB, q_des)

    if wheel_state is None:
        wheel_state = {
            "rw_x": 0.0,
            "rw_y": 0.0,
            "rw_z": 0.0,
            "rw_x_dot": 0.0,
            "rw_y_dot": 0.0,
            "rw_z_dot": 0.0,
        }

    name_to_value = {
        # -------------------------
        # translation position
        # -------------------------
        "p_x": pW[0], "p_y": pW[1], "p_z": pW[2],
        "px":  pW[0], "py":  pW[1], "pz":  pW[2],
        "x":   pW[0], "y":   pW[1], "z":   pW[2],

        # -------------------------
        # translation velocity
        # -------------------------
        "v_x": vW[0], "v_y": vW[1], "v_z": vW[2],
        "vx":  vW[0], "vy":  vW[1], "vz":  vW[2],

        # -------------------------
        # local attitude error coordinates
        # -------------------------
        # e_x style
        "e_x": att_err[0], "e_y": att_err[1], "e_z": att_err[2],
        "ex":  att_err[0], "ey":  att_err[1], "ez":  att_err[2],

        # e_rx style
        "e_rx": att_err[0], "e_ry": att_err[1], "e_rz": att_err[2],
        "erx":  att_err[0], "ery":  att_err[1], "erz":  att_err[2],

        # dtheta style
        "dtheta_x": att_err[0], "dtheta_y": att_err[1], "dtheta_z": att_err[2],
        "dthetax":  att_err[0], "dthetay":  att_err[1], "dthetaz":  att_err[2],

        # other common local-angle aliases
        "phi_x": att_err[0], "phi_y": att_err[1], "phi_z": att_err[2],
        "delta_theta_x": att_err[0], "delta_theta_y": att_err[1], "delta_theta_z": att_err[2],
        "theta_err_x":   att_err[0], "theta_err_y":   att_err[1], "theta_err_z":   att_err[2],

        # legacy aliases sometimes used for local error vector
        "qx": att_err[0], "qy": att_err[1], "qz": att_err[2],

        # -------------------------
        # body angular velocity
        # -------------------------
        "omega_x": wB[0], "omega_y": wB[1], "omega_z": wB[2],
        "w_x":     wB[0], "w_y":     wB[1], "w_z":     wB[2],
        "wx":      wB[0], "wy":      wB[1], "wz":      wB[2],

        # -------------------------
        # reaction wheel internal states
        # -------------------------
        "rw_x":      wheel_state["rw_x"],
        "rw_y":      wheel_state["rw_y"],
        "rw_z":      wheel_state["rw_z"],
        "rw_x_dot":  wheel_state["rw_x_dot"],
        "rw_y_dot":  wheel_state["rw_y_dot"],
        "rw_z_dot":  wheel_state["rw_z_dot"],
        "rw_x_rate": wheel_state["rw_x_dot"],
        "rw_y_rate": wheel_state["rw_y_dot"],
        "rw_z_rate": wheel_state["rw_z_dot"],
    }

    x_phys = []
    for name in state_order:
        if name not in name_to_value:
            raise RuntimeError(f"Unsupported state name in snapshot: {name}")
        x_phys.append(name_to_value[name])

    return np.array(x_phys, dtype=float)

def build_actuator_command_from_snapshot_input(input_order, u_eq, delta_u, act_map, R_WB=None):
    """
    Supports:
    1) actuator-level snapshot inputs, including aliases such as:
       [thruster_px, ..., rw_z]
       [u_thruster_px, ..., u_rw_z]
       [u_tpx, ..., tau_rw_z]
    2) generalized wrench-level snapshot inputs:
       [F_x, F_y, F_z, tau_x, tau_y, tau_z]
    """
    u_phys = u_eq + delta_u

    # Canonical actuator alias map
    alias_to_actuator = {
        # thrusters
        "thruster_px": "thruster_px",
        "thruster_nx": "thruster_nx",
        "thruster_py": "thruster_py",
        "thruster_ny": "thruster_ny",
        "thruster_pz": "thruster_pz",
        "thruster_nz": "thruster_nz",

        "u_thruster_px": "thruster_px",
        "u_thruster_nx": "thruster_nx",
        "u_thruster_py": "thruster_py",
        "u_thruster_ny": "thruster_ny",
        "u_thruster_pz": "thruster_pz",
        "u_thruster_nz": "thruster_nz",

        "T_px": "thruster_px",
        "T_nx": "thruster_nx",
        "T_py": "thruster_py",
        "T_ny": "thruster_ny",
        "T_pz": "thruster_pz",
        "T_nz": "thruster_nz",
        "tau_rw_x": "rw_x",
        "tau_rw_y": "rw_y",
        "tau_rw_z": "rw_z",

        # compact thruster aliases from newer snapshots
        "u_tpx": "thruster_px",
        "u_tnx": "thruster_nx",
        "u_tpy": "thruster_py",
        "u_tny": "thruster_ny",
        "u_tpz": "thruster_pz",
        "u_tnz": "thruster_nz",

        # reaction wheels
        "rw_x": "rw_x",
        "rw_y": "rw_y",
        "rw_z": "rw_z",

        "u_rw_x": "rw_x",
        "u_rw_y": "rw_y",
        "u_rw_z": "rw_z",

        # newer torque aliases
        "tau_rw_x": "rw_x",
        "tau_rw_y": "rw_y",
        "tau_rw_z": "rw_z",
    }

    # Case 1: actuator-level input names
    if all(name in alias_to_actuator for name in input_order):
        u = np.zeros(len(act_map), dtype=float)

        for i, name in enumerate(input_order):
            mapped_name = alias_to_actuator[name]
            if mapped_name not in act_map:
                raise RuntimeError(f"Actuator name {mapped_name} not found in MuJoCo model.")
            aid = act_map[mapped_name]
            if aid >= 0:
                u[aid] = u_phys[i]

        return clamp_actuator_commands(u)

    # Case 2: generalized wrench-level
    generic_names = {"F_x", "F_y", "F_z", "tau_x", "tau_y", "tau_z"}
    if all(name in generic_names for name in input_order):
        u = np.zeros(len(act_map), dtype=float)

        name_to_value = {name: val for name, val in zip(input_order, u_phys)}
        F_W = np.array([
            name_to_value.get("F_x", 0.0),
            name_to_value.get("F_y", 0.0),
            name_to_value.get("F_z", 0.0),
        ], dtype=float)

        tauB = np.array([
            name_to_value.get("tau_x", 0.0),
            name_to_value.get("tau_y", 0.0),
            name_to_value.get("tau_z", 0.0),
        ], dtype=float)

        if R_WB is None:
            raise RuntimeError("R_WB required for generalized-force mapping.")
        F_B = R_WB.T @ F_W

        set_axis_with_min(u, act_map, "thruster_px", "thruster_nx", F_B[0], Fsat=20.0)
        set_axis_with_min(u, act_map, "thruster_py", "thruster_ny", F_B[1], Fsat=20.0)
        set_axis_with_min(u, act_map, "thruster_pz", "thruster_nz", F_B[2], Fsat=20.0)

        if act_map["rw_x"] >= 0:
            u[act_map["rw_x"]] = tauB[0]
        if act_map["rw_y"] >= 0:
            u[act_map["rw_y"]] = tauB[1]
        if act_map["rw_z"] >= 0:
            u[act_map["rw_z"]] = tauB[2]

        return clamp_actuator_commands(u)

    raise RuntimeError(f"Unsupported input_order from snapshot: {input_order}")

# =========================
# Main controller
# =========================
def lqr_controller(
    model,
    data,
    lqr_info,
    robot_qpos_adr,
    robot_qvel_adr,
    rw_qpos_adr,
    rw_qvel_adr,
    act_map,
    desired
):
    # Read state
    pW = np.array(data.qpos[robot_qpos_adr:robot_qpos_adr+3], dtype=float)
    qWB = np.array(data.qpos[robot_qpos_adr+3:robot_qpos_adr+7], dtype=float)

    vW = np.array(data.qvel[robot_qvel_adr:robot_qvel_adr+3], dtype=float)
    wW = np.array(data.qvel[robot_qvel_adr+3:robot_qvel_adr+6], dtype=float)

    qWB = qWB / max(np.linalg.norm(qWB), 1e-12)
    if qWB[0] < 0:
        qWB = -qWB

    R_WB = quat_to_R(qWB)
    wB = R_WB.T @ wW

    p_des = np.asarray(desired["p"], dtype=float)
    v_des = np.asarray(desired["v"], dtype=float)
    q_des = np.asarray(desired["q"], dtype=float)
    w_des = np.asarray(desired["w"], dtype=float)

    q_des = q_des / max(np.linalg.norm(q_des), 1e-12)
    if q_des[0] < 0:
        q_des = -q_des

    # Build physical state in snapshot order
    wheel_state = {
        "rw_x": data.qpos[rw_qpos_adr["rw_x"]],
        "rw_y": data.qpos[rw_qpos_adr["rw_y"]],
        "rw_z": data.qpos[rw_qpos_adr["rw_z"]],
        "rw_x_dot": data.qvel[rw_qvel_adr["rw_x"]],
        "rw_y_dot": data.qvel[rw_qvel_adr["rw_y"]],
        "rw_z_dot": data.qvel[rw_qvel_adr["rw_z"]],
    }

    x_phys = build_state_vector(
        lqr_info["state_order"],
        pW=pW,
        vW=vW - v_des,
        qWB=qWB,
        wB=wB - w_des,
        q_des=q_des,
        wheel_state=wheel_state,
    )

    # Position states should remain physical positions, not pre-subtracted here
    # because x_eq already contains the target waypoint in the snapshot.
    # So overwrite translation/velocity states consistently:
    name_to_correct_value = {
        "p_x": pW[0], "p_y": pW[1], "p_z": pW[2],
        "px":  pW[0], "py":  pW[1], "pz":  pW[2],
        "x":   pW[0], "y":   pW[1], "z":   pW[2],

        "v_x": vW[0], "v_y": vW[1], "v_z": vW[2],
        "vx":  vW[0], "vy":  vW[1], "vz":  vW[2],
    }
    for i, name in enumerate(lqr_info["state_order"]):
        if name in name_to_correct_value:
            x_phys[i] = name_to_correct_value[name]

    # LQR law
    delta_x = x_phys - lqr_info["x_eq"]
    delta_u = -lqr_info["K"] @ delta_x

    u_cmd = build_actuator_command_from_snapshot_input(
        lqr_info["input_order"],
        lqr_info["u_eq"],
        delta_u,
        act_map,
        R_WB=R_WB,
    )

    print("pW =", pW)
    print("delta_x[:6] =", delta_x[:6])
    print("delta_u[:6] =", delta_u[:6])
    print("u_cmd[:6] =", u_cmd[:6])

    return u_cmd


# =========================
# Main
# =========================
def main():
    viewpoints_pos, viewpoints_vel, viewpoints_quat, viewpoints_omega = load_waypoints(WAYPOINT_JSON)
    lqr_info = load_lqr_from_snapshot(SNAPSHOT_JSON)

    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)

    # Actuator map
    act_map = {}
    for nm in ALL_ACT_NAMES:
        act_map[nm] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)

    thruster_ids = [act_map[nm] for nm in THRUSTER_NAMES]
    burn_time_s = np.zeros(6, dtype=float)
    last_ctrl = np.zeros(model.nu, dtype=float)

    # Robot freejoint addresses
    robot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    if robot_bid < 0 or model.body_jntnum[robot_bid] <= 0:
        raise RuntimeError("Could not find freejoint on body 'robot'.")

    robot_jid = int(model.body_jntadr[robot_bid])
    robot_qpos_adr = int(model.jnt_qposadr[robot_jid])
    robot_qvel_adr = int(model.jnt_dofadr[robot_jid])

    # Reaction wheel joint addresses
    rw_jids = {
        "rw_x": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rw_x"),
        "rw_y": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rw_y"),
        "rw_z": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rw_z"),
    }

    rw_qpos_adr = {}
    rw_qvel_adr = {}

    for name, jid in rw_jids.items():
        if jid < 0:
            raise RuntimeError(f"Could not find joint '{name}' in MuJoCo model.")
        rw_qpos_adr[name] = int(model.jnt_qposadr[jid])
        rw_qvel_adr[name] = int(model.jnt_dofadr[jid])

    # Spawn at first waypoint
    p0 = np.array([8, 8, 4])
    q0 = np.array([1, 0, 0, 0])
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    if q0[0] < 0:
        q0 = -q0

    data.qpos[robot_qpos_adr:robot_qpos_adr+3] = p0
    data.qpos[robot_qpos_adr+3:robot_qpos_adr+7] = q0

    mujoco.mj_forward(model, data)

    vp_idx = 0
    stop_flag = None
    DT = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 20.0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat = [0, 0, 0]
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45

        t0_wall = time.perf_counter()
        t0_sim = data.time
        next_ctrl = t0_wall
        next_render = t0_wall

        while viewer.is_running():
            now = time.perf_counter()

            # Control
            if now >= next_ctrl:
                desired = {
                    "p": np.array([1, 1, 1], dtype=float),
                    "v": np.zeros(3, dtype=float),
                    "q": np.array( [0.9186, 0.1768, 0.3062, 0.1768], dtype=float),
                    "w": np.zeros(3, dtype=float),
                }

                u = lqr_controller(
                    model,
                    data,
                    lqr_info,
                    robot_qpos_adr,
                    robot_qvel_adr,
                    rw_qpos_adr,
                    rw_qvel_adr,
                    act_map,
                    desired
                )

                data.ctrl[:] = u
                last_ctrl = u.copy()

                next_ctrl += 1.0 / CONTROL_HZ
                if now > next_ctrl + 2.0 / CONTROL_HZ:
                    next_ctrl = now + 1.0 / CONTROL_HZ

            # Physics
            target_sim_time = t0_sim + (now - t0_wall) * RUN_SPEED
            steps = step_until(model, data, target_sim_time)

            if steps > 0:
                dt_sum = steps * DT
                for i, aid in enumerate(thruster_ids):
                    if aid >= 0 and last_ctrl[aid] > FIRE_THRESH:
                        burn_time_s[i] += dt_sum

            # Rendering
            if now >= next_render:
                viewer.sync()
                next_render += 1.0 / RENDER_HZ
                if now > next_render + 2.0 / RENDER_HZ:
                    next_render = now + 1.0 / RENDER_HZ

            time.sleep(IDLE_SLEEP)

    print("\n=== Thruster usage summary ===")
    for i, nm in enumerate(THRUSTER_NAMES):
        print(f"{nm:12s} time = {burn_time_s[i]:8.3f} s")


if __name__ == "__main__":
    main()
