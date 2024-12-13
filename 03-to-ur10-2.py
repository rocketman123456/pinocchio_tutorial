import time
import casadi
import example_robot_data as robex
import numpy as np
import pinocchio as pin
import threading
from pinocchio import casadi as cpin
from meshcat_viewer_wrapper import MeshcatVisualizer

# robot arm example
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

    nq = model.nq
    nv = model.nv
    nx = nq + nv
    ndx = 2 * nv
    cx = casadi.SX.sym("x", nx, 1)
    cdx = casadi.SX.sym("dx", nv * 2, 1)
    cq = cx[:nq]
    cv = cx[nq:]
    caq = casadi.SX.sym("a", nv, 1)

    # Compute kinematics casadi graphs
    cpin.forwardKinematics(cmodel, cdata, cq, cv, caq)
    cpin.updateFramePlacements(cmodel, cdata)

    error3_tool = casadi.Function(
        "etool3",
        [cx],
        [cdata.oMf[tool_id].translation - in_world_M_target.translation],
    )
    error6_tool = casadi.Function(
        "etool6",
        [cx],
        [cpin.log6(cdata.oMf[tool_id].inverse() * cpin.SE3(in_world_M_target)).vector],
    )
    error_tool = error6_tool

    T = 50
    DT = 0.002
    w_vel = 0.1
    w_conf = 5
    w_term = 1e4

    cnext = casadi.Function(
        "next",
        [cx, caq],
        [
            casadi.vertcat(
                cpin.integrate(cmodel, cx[:nq], cx[nq:] * DT + caq * DT**2),
                cx[nq:] + caq * DT,
            )
        ],
    )

    opti = casadi.Opti()
    var_xs = [opti.variable(nx) for t in range(T + 1)]
    var_as = [opti.variable(nv) for t in range(T)]
    totalcost = 0

    # running cost
    for t in range(T):
        totalcost += 1e-3 * DT * casadi.sumsqr(var_xs[t][nq:])
        totalcost += 1e-4 * DT * casadi.sumsqr(var_as[t])

    # terminate cost
    totalcost += w_term * casadi.sumsqr(error_tool(var_xs[T]))

    # set constraints
    opti.subject_to(var_xs[0][:nq] == robot.q0)
    opti.subject_to(var_xs[0][nq:] == 0)
    for t in range(T):
        opti.subject_to(cnext(var_xs[t], var_as[t]) == var_xs[t + 1])

    # set cost function
    opti.minimize(totalcost)
    opti.solver("ipopt")  # set numerical backend
    # opti.callback(lambda i: displayScene(opti.debug.value(var_qs[-1])))

    # Caution: in case the solver does not converge, we are picking the candidate values
    # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
    try:
        sol = opti.solve_limited()
        sol_xs = [opti.value(var_x) for var_x in var_xs]
        sol_as = [opti.value(var_a) for var_a in var_as]
    except:
        print("ERROR in convergence, plotting debug info.")
        sol_xs = [opti.debug.value(var_x) for var_x in var_xs]
        sol_as = [opti.debug.value(var_a) for var_a in var_as]

    q = [x[:nq] for x in sol_xs][T]
    robot.q0 = q
    displayTraj([x[:nq] for x in sol_xs], 1e-1)
    time.sleep(0.05)
