import time
import casadi
import example_robot_data as robex
import numpy as np
import pinocchio as pin
import threading
from pinocchio import casadi as cpin
from meshcat_viewer_wrapper import MeshcatVisualizer

robot = robex.load("ur10")

# target
in_world_M_target = pin.SE3(
    pin.utils.rotate("y", 3),
    np.array([-0.5, 0.1, 0.2]),
)  # x,y,z

q = np.array([0, -3.14 / 2, 0, 0, 0, 0])
tool_frameName = "tool0"
robot.q0 = q

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
tool_id = model.getFrameId(tool_frameName)

# Open the viewer
viz = MeshcatVisualizer(robot, url=None)
viz.display(robot.q0)
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [0.05, 0.1, 0.2], [1.0, 0.2, 0.2, 0.5])
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [0.08] * 3, [0.2, 0.2, 1.0, 0.5])

lock = threading.Lock()


def displayScene(q, dt=1e-1):
    """
    Given the robot configuration, display:
    - the robot
    - a box representing tool_id
    - a box representing in_world_M_target
    """
    pin.framesForwardKinematics(model, data, q)
    M = data.oMf[tool_id]
    viz.applyConfiguration(boxID, in_world_M_target)
    viz.applyConfiguration(tipID, M)
    viz.display(q)
    time.sleep(dt)


def displayTraj(qs, dt=1e-2):
    for q in qs[1:]:
        displayScene(q, dt=dt)


# Function to update the visualizer in a separate thread
def visualizer_thread():
    while True:
        with lock:
            viz.display(q)  # Update the visualizer with the latest configuration
        time.sleep(0.01)  # Adjust for desired update rate


# Start the visualizer thread
thread = threading.Thread(target=visualizer_thread, daemon=True)
thread.start()

curr_time = 0.0

while True:
    # target
    delta_x = 0.1 * np.sin(curr_time)
    delta_y = 0.1 * np.sin(curr_time)
    delta_z = 0.2 * np.sin(curr_time)
    in_world_M_target = pin.SE3(
        pin.utils.rotate("y", 3),
        np.array([-0.5, 0.1, 0.2 + delta_z]),
    )  # x,y,z
    curr_time += 0.3

    # --- Casadi helpers
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    # Compute kinematics casadi graphs
    cq = casadi.SX.sym("x", model.nq, 1)
    cpin.framesForwardKinematics(cmodel, cdata, cq)

    error3_tool = casadi.Function(
        "etool3",
        [cq],
        [cdata.oMf[tool_id].translation - in_world_M_target.translation],
    )
    error6_tool = casadi.Function(
        "etool6",
        [cq],
        [cpin.log6(cdata.oMf[tool_id].inverse() * cpin.SE3(in_world_M_target)).vector],
    )
    error_tool = error6_tool

    T = 10
    w_run = 0.1
    w_term = 1

    opti = casadi.Opti()
    var_qs = [opti.variable(model.nq) for t in range(T + 1)]
    totalcost = 0

    # set init value
    # for i in range(T):
    #     opti.set_initial(var_qs[i], q)
    # for i in range(T):
    #     opti.set_initial(var_xs[i], q)

    # running cost
    for t in range(T):
        totalcost += w_run * casadi.sumsqr(var_qs[t] - var_qs[t + 1])

    # terminate cost
    totalcost += w_term * casadi.sumsqr(error_tool(var_qs[T]))

    # set constraints
    opti.subject_to(var_qs[0] == robot.q0)

    # set cost function
    opti.minimize(totalcost)
    opti.solver("ipopt")  # set numerical backend
    # opti.callback(lambda i: displayScene(opti.debug.value(var_qs[-1])))

    # Caution: in case the solver does not converge, we are picking the candidate values
    # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
    try:
        sol = opti.solve_limited()
        sol_qs = [opti.value(var_q) for var_q in var_qs]
    except:
        print("ERROR in convergence, plotting debug info.")
        sol_qs = [opti.debug.value(var_q) for var_q in var_qs]

    with lock:
        q = sol_qs[T]
        robot.q0 = q
        displayTraj(sol_qs, 1e-1)
    time.sleep(0.05)
