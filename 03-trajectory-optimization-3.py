import time
import casadi
import example_robot_data as robex
import numpy as np
import pinocchio as pin
import threading
import matplotlib.pylab as plt
from pinocchio import casadi as cpin
from meshcat_viewer_wrapper import MeshcatVisualizer
from types import SimpleNamespace


robot = robex.load("talos_legs")
# Open the viewer
viz = MeshcatVisualizer(robot, url=None)
viz.display(robot.q0)

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()

in_world_M_target = pin.SE3(pin.utils.rotate("y", 3), np.array([-0.1, 0.2, 0.45094]))  # x,y,z
contacts = [SimpleNamespace(name="left_sole_link", type=pin.ContactType.CONTACT_6D)]
tool_frameName = "right_sole_link"

tool_id = model.getFrameId(tool_frameName)
for c in contacts:
    c.id = model.getFrameId(c.name)
    assert c.id < len(model.frames)

# --- Add box to represent target
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [0.05, 0.1, 0.2], [1.0, 0.2, 0.2, 0.5])
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [0.08] * 3, [0.2, 0.2, 1.0, 0.5])
for c in contacts:
    c.viz = f"world/contact_{c.name}"
    viz.addSphere(c.viz, [0.07], [0.8, 0.8, 0.2, 0.5])


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
    for c in contacts:
        viz.applyConfiguration(c.viz, data.oMf[c.id])
    viz.display(q)
    time.sleep(dt)


def displayTraj(qs, dt=1e-2):
    for q in qs[1:]:
        displayScene(q, dt=dt)


T = 50
DT = 0.002
w_vel = 0.1
w_conf = 5
w_term = 1e4

while True:
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

    # Sym graph for the integration operation x,dx -> x(+)dx = [model.integrate(q,dq),v+dv]
    cintegrate = casadi.Function(
        "integrate",
        [cx, cdx],
        [casadi.vertcat(cpin.integrate(cmodel, cx[:nq], cdx[:nv]), cx[nq:] + cdx[nv:])],
    )

    # Sym graph for the operational error
    error_tool = casadi.Function(
        "etool3",
        [cx],
        [cdata.oMf[tool_id].translation - in_world_M_target.translation],
    )

    # Sym graph for the integration operation x' = [ q+vDT+aDT**2, v+aDT ]
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

    # Sym graph for the contact constraint and Baugart correction terms
    # Works for both 3D and 6D contacts.
    # Uses the contact list <contacts> where each item must have a <name>, an <id> and a <type> field.
    dpcontacts = {}  # Error in contact position
    vcontacts = {}  # Error in contact velocity
    acontacts = {}  # Contact acceleration

    for c in contacts:
        if c.type == pin.ContactType.CONTACT_3D:
            p0 = data.oMf[c.id].translation.copy()
            dpcontacts[c.name] = casadi.Function(
                f"dpcontact_{c.name}",
                [cx],
                [-(cdata.oMf[c.id].inverse().act(casadi.SX(p0)))],
            )
            vcontacts[c.name] = casadi.Function(
                f"vcontact_{c.name}",
                [cx],
                [cpin.getFrameVelocity(cmodel, cdata, c.id, pin.LOCAL).linear],
            )
            acontacts[c.name] = casadi.Function(
                f"acontact_{c.name}",
                [cx, caq],
                [cpin.getFrameClassicalAcceleration(cmodel, cdata, c.id, pin.LOCAL).linear],
            )
        elif c.type == pin.ContactType.CONTACT_6D:
            p0 = data.oMf[c.id]
            dpcontacts[c.name] = casadi.Function(f"dpcontact_{c.name}", [cx], [np.zeros(6)])
            vcontacts[c.name] = casadi.Function(
                f"vcontact_{c.name}",
                [cx],
                [cpin.getFrameVelocity(cmodel, cdata, c.id, pin.LOCAL).vector],
            )
            acontacts[c.name] = casadi.Function(
                f"acontact_{c.name}",
                [cx, caq],
                [cpin.getFrameAcceleration(cmodel, cdata, c.id, pin.LOCAL).vector],
            )

    Kp = 200
    Kv = 2 * np.sqrt(Kp)

    # Get initial contact position (for Baumgart correction)
    pin.framesForwardKinematics(model, data, robot.q0)

    cbaumgart = {c.name: casadi.Function(f"K_{c.name}", [cx], [Kp * dpcontacts[c.name](cx) + Kv * vcontacts[c.name](cx)]) for c in contacts}

    opti = casadi.Opti()
    var_dxs = [opti.variable(ndx) for t in range(T + 1)]
    var_as = [opti.variable(nv) for t in range(T)]
    var_xs = [cintegrate(np.concatenate([robot.q0, np.zeros(nv)]), var_dx) for var_dx in var_dxs]

    totalcost = 0
    # Define the running cost
    for t in range(T):
        totalcost += 1e-3 * DT * casadi.sumsqr(var_xs[t][nq:])
        totalcost += 1e-4 * DT * casadi.sumsqr(var_as[t])
    totalcost += 1e4 * casadi.sumsqr(error_tool(var_xs[T]))

    opti.subject_to(var_xs[0][:nq] == robot.q0)
    opti.subject_to(var_xs[0][nq:] == 0)

    for t in range(T):
        opti.subject_to(cnext(var_xs[t], var_as[t]) == var_xs[t + 1])

    for t in range(T):
        for c in contacts:
            # correction = Kv* vcontacts[c.name](var_xs[t]) + Kp * dpcontacts[c.name](var_xs[t])
            correction = cbaumgart[c.name](var_xs[t])
            opti.subject_to(acontacts[c.name](var_xs[t], var_as[t]) == -correction)

    ### SOLVE
    opti.minimize(totalcost)
    opti.solver("ipopt")  # set numerical backend
    # opti.callback(lambda i: displayScene(opti.debug.value(var_xs[-1][:nq])))

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
    # displayScene(robot.q0)
    displayScene(robot.q0, 1)
    displayTraj([x[:nq] for x in sol_xs], DT)

    # Plotting the contact gaps
    h_pcontacts = []
    h_vcontacts = []
    h_acontacts = []
    for t in range(T):
        x = sol_xs[t]
        q = x[:nq]
        v = x[nq:]
        a = sol_as[t]
        h_pcontacts.append(np.concatenate([opti.value(dpcontacts[c.name](var_xs[t])) for c in contacts]))
        h_vcontacts.append(np.concatenate([opti.value(vcontacts[c.name](var_xs[t])) for c in contacts]))
        h_acontacts.append(np.concatenate([opti.value(acontacts[c.name](var_xs[t], var_as[t])) for c in contacts]))

    # plt.ion()
    # fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    # ax[0].plot(h_pcontacts)
    # ax[0].set_title("delta position")
    # ax[0].axis((-2.45, 51.45, -0.5e-3, 0.5e-3))
    # ax[1].plot(h_vcontacts)
    # ax[1].set_title("velocity")
    # ax[1].axis((-2.45, 51.45, -0.006627568040194312, 0.007463128239663308))
