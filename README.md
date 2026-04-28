# Auto-Controller

**Auto-Controller** is a semester-long demo project for **agentic control-system design**. The goal is to let a user describe a control task in natural language, then automatically produce a structured plant model, a controller design, and an overall workflow history snapshot.

At the current stage, the project focuses on a small but stable set of benchmark problems rather than trying to support every possible control system. It currently supports:

- **Single pendulum** equilibrium stabilization around arbitrary local operating points.
- **Mass-spring-damper** equilibrium stabilization around arbitrary local operating points.

- **Local LQR stabilization for a MuJoCo free-floating space robot** described by `robot.xml`, under small-angle / local-linearization assumptions near the target state.

The project is intended as a proof-of-concept showing how LLMs, structured schemas, and deterministic control solvers can be connected into one automated control-design workflow.

---

## Demo Overview

The user provides a natural-language control-design request. For the MuJoCo space-robot case, the user also uploads the prepared `robot.xml` file through the Gradio interface. The system then runs a multi-agent pipeline:

```text
User request
    |
    v
Strategy Agent
    |
    v
Strategy QC Agent
    |
    v
Modeling Agent
    |
    v
Modeling QC Agent
    |
    v
Control Agent
    |
    v
Control QC Agent
    |
    v
Structured controller snapshot
```

The generated output includes:

- Parsed task specification.
- Selected control strategy, such as `PD` or `LQR`.
- Structured nonlinear and/or local linear plant model.
- Operating point and deviation-coordinate definitions.
- Controller gains and implementation notes.
- QC reports from each review stage.
- A saved `controller_snapshot.json` file used by the simulation scripts.

---

## Repository Structure

```text
Auto-Controller/
├── app/
│   ├── agents/
│   │   ├── strategy_agent.py
│   │   ├── strategy_qc_agent.py
│   │   ├── modeling_agent.py
│   │   ├── modeling_qc_agent.py
│   │   ├── control_agent.py
│   │   └── control_qc_agent.py
│   ├── control/
│   │   ├── lqr_solver.py
│   │   └── pid_tuning.py
│   ├── schemas/
│   │   ├── common.py
│   │   ├── modeling.py
│   │   └── control.py
│   ├── config.py
│   └── graph.py
├── app_gradio.py
├── simulate_controller.py
├── autocontroller.py
├── robot.xml
├── controller_snapshot.json
└── .gitignore
```

### Key files

| File | Purpose |
|---|---|
| `app_gradio.py` | Gradio user interface for entering control-design prompts and uploading optional XML / URDF files. |
| `app/graph.py` | Defines the LangGraph workflow connecting strategy, modeling, control, and QC agents. |
| `app/agents/` | LLM-based agents and reviewer agents. |
| `app/schemas/` | Pydantic schemas for structured model, controller, and QC outputs. |
| `app/control/lqr_solver.py` | Deterministic continuous LQR solver using Riccati equations. |
| `app/control/pid_tuning.py` | Deterministic PD gain generation from second-order local models. |
| `simulate_controller.py` | Benchmark simulation for pendulum and mass-spring-damper systems. |
| `autocontroller.py` | MuJoCo-based local LQR controller runner for the free-floating space robot. |
| `robot.xml` | MuJoCo model for the space robot benchmark. |
| `controller_snapshot.json` | Example generated controller package. |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YunhaoTsao10/Auto-Controller.git
cd Auto-Controller
```

### 2. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

The repository does not currently include a `requirements.txt`, so install the main dependencies manually:

```bash
pip install openai python-dotenv pydantic langgraph gradio numpy scipy matplotlib mujoco
```

If you only want to run the Gradio design interface or the pendulum / mass-spring-damper benchmarks, `mujoco` is optional. The space-robot benchmark requires a working MuJoCo Python installation.

---

## API Configuration

This project uses the OpenAI Python client. The API key should be provided through an environment variable or a local `.env` file.

### Option A: Use a `.env` file

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

The code already calls `load_dotenv()` in `app/config.py`, so the key will be loaded automatically when the app starts.

### Option B: Export the key in your shell

macOS / Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

### Model setting

The model is configured in:

```python
# app/config.py
OPENAI_MODEL = "gpt-5.4"
```

If this model name is not available in your environment, replace it with a model available to your OpenAI account.

---

## Running the Gradio App

Start the web interface with:

```bash
python app_gradio.py
```

Then open the local Gradio URL shown in the terminal.

The interface provides:

- A text box for the natural-language control-design instruction.
- An optional XML / URDF upload field.
- A structured JSON output panel.
- A rendered dynamics-model panel.
- A rendered controller-design panel.

After a successful run, the app writes:

```text
controller_snapshot.json
```

This snapshot can be reused by the benchmark simulation scripts.

---

## Example Prompts

### Single Pendulum, PD Control

```text
Please model a simple pendulum and design a local PD stabilizing controller around theta = 2.5 rad.
Use m = 1 kg, L = 1 m, g = 9.81 m/s^2, and damping b = 0.05.
Target less than 1% overshoot and settling time around 1 seconds.
```

### Mass-Spring-Damper, PD Control

```text
Please model a mass-spring-damper system and design a local stabilizing PD controller around x = 1.0 m.
Use m = 1.0 kg, c = 0.4 N·s/m, k = 4.0 N/m.
Target less than 1% overshoot and settling time around 1 seconds.
```

### Space Robot, Local LQR Control

```text
The uploaded robot.xml describes a free-floating space robot in MuJoCo.
Task:
Design a continuous-time LQR controller for point-to-point motion from a fixed initial state to a fixed goal state.
Requirements:
1. Do not reduce the robot to a single-axis model.
2. Use a local linear state-space model suitable for LQR.
3. The state should include at least position, velocity, attitude, and angular velocity.
4. The control inputs should correspond to the robot actuators in the XML.
5. Reuse the uploaded XML content to infer the robot structure, inertia, and available actuators.
6. The result must provide:
   - operating point
   - state order
   - input order
   - A, B, Q, R matrices
   - equilibrium state x_eq
   - equilibrium input u_eq
   - LQR feedback law u = u_eq - K(x - x_eq)
Control objective:
Move the robot from an initial waypoint to a target waypoint in free space while stabilizing attitude.
Initial desired state:
position = [0, 0, 0]
velocity = [0, 0, 0]
quaternion = [1, 0, 0, 0]
angular_velocity = [0, 0, 0]
Target desired state:
position = [1, 1, 1]
velocity = [0, 0, 0]
quaternion = [0.9186, 0.1768, 0.3062, 0.1768]
angular_velocity = [0, 0, 0]
Important:
- Treat this as a multi-input multi-state spacecraft control problem.
- Do not collapse it into a SISO second-order template.
- If needed, make clearly stated small-angle and local-linearity assumptions near the target state.
- Do not add internal wheel-speed states unless they are modeled consistently and fully in A and B.
```

---

## Running Benchmark Simulations

### Pendulum / Mass-Spring-Damper Simulation

After generating a `controller_snapshot.json` from the Gradio app, run:

```bash
python simulate_controller.py
```

The script will:

1. Load the controller snapshot if available.
2. Infer the plant type and controller parameters.
3. Simulate the closed-loop system.
4. Print tracking metrics.
5. Plot state and control histories.
6. Show an animation for the supported benchmark.

The current benchmark registry includes:

```python
PLANT_REGISTRY = {
    "pendulum": pendulum_dynamics,
    "mass_spring_damper": msd_dynamics,
}
```

This makes it easy to add more systems later, such as a double pendulum or cart-pole.

### MuJoCo Space Robot Simulation

The space-robot benchmark uses a two-stage workflow. The controller is first generated by the agentic design pipeline, and then the generated snapshot is tested in MuJoCo.

#### Prerequisite

Before running this benchmark, make sure that MuJoCo is installed and working in your Python environment

#### Step 1: Generate the controller snapshot from Gradio

Start the Gradio interface:

```bash
python app_gradio.py
```

In the web interface:

1. Enter the space-robot LQR prompt.
2. Upload the prepared `robot.xml` file.
3. Run the agentic control-design pipeline.
4. Confirm that a new `controller_snapshot.json` is generated.

The agent uses the uploaded MuJoCo XML to infer the robot structure, available actuators, state order, input order, local linear model, equilibrium point, and LQR feedback law.

#### Step 2: Test the generated controller in MuJoCo

After the snapshot has been generated, run:

```bash
python autocontroller.py
```

This script loads:

- `robot.xml`
- `controller_snapshot.json`

The controller uses a local LQR law:

```text
u = u_eq - K(x - x_eq)
```

For attitude, the implementation uses a local small-angle error vector derived from the quaternion error. The MuJoCo actuator command is then mapped to thrusters and reaction-wheel torques.

This benchmark should be interpreted as a local stabilization and validation demo, not as a full global spacecraft motion-planning or large-angle attitude-control solution.

---

## Benchmark GIFs

| Benchmark | Controller |  Demo |
|---|---|---|
| Single Pendulum | PD | <img width="384" height="240" alt="1" src="https://github.com/user-attachments/assets/e89bd92e-e3dd-4671-a4e6-84c97c0e2c54" />|
| Single Pendulum | LQR | <img width="384" height="240" alt="2" src="https://github.com/user-attachments/assets/0fedc403-2f77-4e9e-80ed-553e1ccd1625" />|
| Mass-Spring-Damper | PD | <img width="384" height="240" alt="3" src="https://github.com/user-attachments/assets/e89110c5-14a2-4289-b8c5-0234e13c731a" />|
| Mass-Spring-Damper | LQR |<img width="384" height="240" alt="4" src="https://github.com/user-attachments/assets/3c9ab3de-d90b-4df9-89f2-34c6a6bba8ea" />|
| Space Robot | LQR |<img width="384" height="240" alt="5" src="https://github.com/user-attachments/assets/a3651b78-bf62-494a-91be-8b01eaf448b9" />|

---

## Current Supported Scope

This repository is intentionally focused on a small, reliable set of cases.

### Supported now

- Natural-language-to-structured-control-design workflow.
- PD/PID-family design for second-order SISO mechanical systems.
- Continuous-time LQR for controller-ready local linear models.
- Single pendulum benchmark.
- Mass-spring-damper benchmark.
- MuJoCo free-floating robot local LQR demo under small-angle assumptions.
- Agent-level QC and retry logic.

### Not fully supported yet

- Global nonlinear control guarantees.
- Large-angle global attitude control for the space robot.
- Robust control, MPC, CBF, adaptive control, or learning-based control.

- Fully automated benchmark generation for arbitrary plants.

---

## Design Philosophy

This project separates the workflow into two parts:

1. **LLM-assisted reasoning and packaging**
   - Parse user intent.
   - Select a controller family.
   - Produce a structured model.
   - Write operating-point assumptions and implementation notes.

2. **Deterministic control computation**
   - Solve LQR gains using Riccati equations.
   - Compute PD gains from a second-order local model.
   - Validate matrix dimensions and state/input order.
   - Export a reusable controller snapshot.

This separation is important because the LLM is used for flexible modeling and interface generation, while the final numerical controller synthesis is handled by deterministic solvers whenever possible.



