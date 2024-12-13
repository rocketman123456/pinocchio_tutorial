import time
import casadi
import example_robot_data as robex
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from meshcat_viewer_wrapper import MeshcatVisualizer
from types import SimpleNamespace

# talos standing with contacts in older version
robot = robex.load("talos_legs")
# Open the viewer
viz = MeshcatVisualizer(robot, url=None)
viz.display(robot.q0)

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()

Mtarget = pin.SE3(pin.utils.rotate("y", 3), np.array([-0.1, 0.2, 0.45094]))  # x,y,z
contacts = [SimpleNamespace(name="left_sole_link", type=pin.ContactType.CONTACT_6D)]
endEffectorFrameName = "right_sole_link"

endEffector_ID = model.getFrameId(endEffectorFrameName)
for c in contacts:
    c.id = model.getFrameId(c.name)
    assert c.id < len(model.frames)
    c.jid = model.frames[c.id].parentJoint
    c.placement = model.frames[c.id].placement
    c.model = pin.RigidConstraintModel(c.type, model, c.jid, c.placement)
contact_models = [c.model for c in contacts]

# Baumgart correction
Kv = 20.0
Kp = 0.0
# Tuning of the proximal solver (minimal version)
prox_settings = pin.ProximalSettings(0, 1e-6, 1)

contact_datas = [c.createData() for c in contact_models]
# for c in contact_models:
#     c.corrector.Kd = Kv
#     c.corrector.Kp = Kp

pin.initConstraintDynamics(model, data, contact_models)
q = robot.q0.copy()
v = np.zeros(model.nv)
tau = np.zeros(model.nv)
pin.constraintDynamics(model, data, q, v, tau, contact_models, contact_datas)

T = 50
DT = 0.002
w_vel = 0.1
w_conf = 5
w_term = 1e4

# --- Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()
ccontact_models = [cpin.RigidConstraintModel(c) for c in contact_models]
ccontact_datas = [c.createData() for c in ccontact_models]
cprox_settings = cpin.ProximalSettings(prox_settings.absolute_accuracy, prox_settings.mu, prox_settings.max_iter)
cpin.initConstraintDynamics(cmodel, cdata, ccontact_models)

nq = model.nq
nv = model.nv
nx = nq + nv
ndx = 2 * nv
cx = casadi.SX.sym("x", nx, 1)
cdx = casadi.SX.sym("dx", nv * 2, 1)
cq = cx[:nq]
cv = cx[nq:]
caq = casadi.SX.sym("a", nv, 1)
ctauq = casadi.SX.sym("tau", nv, 1)

# Compute kinematics casadi graphs
cpin.constraintDynamics(cmodel, cdata, cq, cv, ctauq, ccontact_models, ccontact_datas)
cpin.forwardKinematics(cmodel, cdata, cq, cv, caq)
cpin.updateFramePlacements(cmodel, cdata)

# Sym graph for the integration operation x,dx -> x(+)dx = [model.integrate(q,dq),v+dv]
cintegrate = casadi.Function(
    "integrate",
    [cx, cdx],
    [casadi.vertcat(cpin.integrate(cmodel, cx[:nq], cdx[:nv]), cx[nq:] + cdx[nv:])],
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

# Sym graph for the aba operation
caba = casadi.Function(
    "fdyn",
    [cx, ctauq],
    [cdata.ddq],
)

# Sym graph for the operational error
error_tool = casadi.Function(
    "etool3",
    [cx],
    [cdata.oMf[endEffector_ID].translation - Mtarget.translation],
)

# OCP
opti = casadi.Opti()
var_dxs = [opti.variable(ndx) for t in range(T + 1)]
var_as = [opti.variable(nv) for t in range(T)]
var_us = [opti.variable(nv - 6) for t in range(T)]
var_xs = [cintegrate(np.concatenate([robot.q0, np.zeros(nv)]), var_dx) for var_dx in var_dxs]

totalcost = 0
# Define the running cost
for t in range(T):
    totalcost += 1e-3 * DT * casadi.sumsqr(var_xs[t][nq:])
    totalcost += 1e-4 * DT * casadi.sumsqr(var_as[t])
totalcost += 1e1 * casadi.sumsqr(error_tool(var_xs[T]))

opti.subject_to(var_xs[0][:nq] == robot.q0)
opti.subject_to(var_xs[0][nq:] == 0)  # zero initial velocity
opti.subject_to(var_xs[T][nq:] == 0)  # zero terminal velocity

for t in range(T):
    tau = casadi.vertcat(np.zeros(6), var_us[t])
    opti.subject_to(caba(var_xs[t], tau) == var_as[t])
    opti.subject_to(cnext(var_xs[t], var_as[t]) == var_xs[t + 1])

### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt")  # set numerical backend
opti.callback(lambda i: displayScene(opti.debug.value(var_xs[-1][:nq])))

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_xs = [opti.value(var_x) for var_x in var_xs]
    sol_as = [opti.value(var_a) for var_a in var_as]
    sol_us = [opti.value(var_u) for var_u in var_us]
except:
    print("ERROR in convergence, plotting debug info.")
    sol_xs = [opti.debug.value(var_x) for var_x in var_xs]
    sol_as = [opti.debug.value(var_a) for var_a in var_as]
    sol_us = [opti.debug.value(var_u) for var_u in var_us]


def displayScene(q, dt=1e-1):
    """
    Given the robot configuration, display:
    - the robot
    - a box representing tool_id
    - a box representing in_world_M_target
    """
    pin.framesForwardKinematics(model, data, q)
    # viz.applyConfiguration(box1ID, in_world_Base_start)
    # viz.applyConfiguration(box2ID, in_world_Base_target)
    # TODO : add foot air target
    # M = data.oMf[tool_id]
    # viz.applyConfiguration(tipID, M)
    for c in contacts:
        viz.applyConfiguration(c.viz, data.oMf[c.id])
    viz.display(q)
    time.sleep(dt)


def displayTraj(qs, dt=1e-2):
    for q in qs[1:]:
        displayScene(q, dt=dt)


while True:
    displayScene(robot.q0, 1)
    displayTraj([x[:nq] for x in sol_xs], DT)
