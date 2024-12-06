import time
import unittest
import threading
import example_robot_data
import numpy as np
import casadi
import pinocchio as pin
import pinocchio.casadi as cpin
from meshcat_viewer_wrapper import MeshcatVisualizer, colors

robot = example_robot_data.load("ur10")
model = robot.model
data = robot.data

viz = MeshcatVisualizer(robot, url=None)
viz.display(robot.q0)

robot.q0 = np.array([0, -np.pi / 2, 0, 0, 0, 0])
q = robot.q0

tool_id = model.getFrameId("tool0")

in_world_M_target = pin.SE3(
    pin.utils.rotate("x", np.pi / 4),
    np.array([-0.5, 0.1, 0.2]),
)

# --- Add box to represent target

# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [0.05, 0.1, 0.2], [1.0, 0.2, 0.2, 0.5])

# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [0.08] * 3, [0.2, 0.2, 1.0, 0.5])

lock = threading.Lock()


def displayScene(q):
    """
    Given the robot configuration, display:
    - the robot
    - a box representing tool_id
    - a box representing in_world_M_target
    """
    pin.framesForwardKinematics(model, data, q)
    in_world_M_tool = data.oMf[tool_id]
    viz.applyConfiguration(boxID, in_world_M_target)
    viz.applyConfiguration(tipID, in_world_M_tool)
    viz.display(q)
    time.sleep(1e-1)


# Function to update the visualizer in a separate thread
def visualizer_thread():
    while True:
        with lock:
            viz.display(q)  # Update the visualizer with the latest configuration
        time.sleep(0.05)  # Adjust for desired update rate


# --- Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()
cq = casadi.SX.sym("q", model.nq, 1)
# cpin.framesForwardKinematics(cmodel, cdata, cq)

# Start the visualizer thread
thread = threading.Thread(target=visualizer_thread, daemon=True)
thread.start()

curr_time = 0.0

while True:
    # set target pos
    delta_x = 0.1 * np.sin(curr_time)
    delta_y = 0.1 * np.sin(curr_time)
    in_world_M_target = pin.SE3(
        pin.utils.rotate("x", np.pi / 4),
        np.array([-0.5, 0.1 + delta_y, 0.2]),
    )
    curr_time += 0.1

    # --- Casadi helpers
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    cq = casadi.SX.sym("q", model.nq, 1)
    cpin.framesForwardKinematics(cmodel, cdata, cq)

    # setup optimization
    error_tool = casadi.Function(
        "etool",
        [cq],
        [cpin.log6(cdata.oMf[tool_id].inverse() * cpin.SE3(in_world_M_target)).vector],
    )
    opti = casadi.Opti()
    var_q = opti.variable(model.nq)
    opti.set_initial(var_q, q)
    totalcost = casadi.sumsqr(error_tool(var_q))
    opti.minimize(totalcost)
    opti.solver("ipopt")  # select the backend solver
    opti.callback(lambda i: displayScene(opti.debug.value(var_q)))

    # calculate a solution
    try:
        sol = opti.solve_limited()
        sol_q = opti.value(var_q)
    except:
        print("ERROR in convergence, plotting debug info.")
        sol_q = opti.debug.value(var_q)

    pin.framesForwardKinematics(model, data, sol_q)
    opti.value(error_tool(var_q)), pin.log(data.oMf[tool_id].inverse() * in_world_M_target).vector

    with lock:
        q = sol_q
        # robot.q0 = q
        displayScene(sol_q)
    time.sleep(0.05)
