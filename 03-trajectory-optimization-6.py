import time
import casadi
import example_robot_data as robex
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from meshcat_viewer_wrapper import MeshcatVisualizer

### Quadruped Jumping Example
robot = robex.load("solo12")
# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()

# Open the viewer
viz = MeshcatVisualizer(robot, url=None)
viz.display(robot.q0)

cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()

q0 = robot.q0
nq = cmodel.nq
nDq = cmodel.nv

# in_world_M_target = pin.SE3(pin.utils.rotate("y", 3), np.array([-0.1, 0.2, 0.45094]))  # x,y,z
# contacts = [SimpleNamespace(name="left_sole_link", type=pin.ContactType.CONTACT_6D)]
# tool_frameName = "right_sole_link"

cq = casadi.SX.sym("cq", nq, 1)
cDq = casadi.SX.sym("cx", nDq, 1)
R = casadi.SX.sym("R", 3, 3)
R_ref = casadi.SX.sym("R_ref", 3, 3)

# Get the index of the frames which are going to be used
IDX_BASE = cmodel.getFrameId("torso_2_link")
IDX_LF = cmodel.getFrameId("leg_left_6_link")
IDX_RF = cmodel.getFrameId("leg_right_6_link")
IDX_LG = cmodel.getFrameId("gripper_left_base_link")
IDX_RG = cmodel.getFrameId("gripper_right_base_link")
IDX_LE = cmodel.getFrameId("arm_left_4_joint")
IDX_RE = cmodel.getFrameId("arm_right_4_joint")

# This is used in order to go from a configuration and the displacement to the final configuration.
# Why pinocchio.integrate and not simply q = q0 + v*dt?
# q and v have different dimensions because q contains quaterniions and this can't be done
# So pinocchio.integrate(config, Dq)
integrate = casadi.Function("integrate", [cq, cDq], [cpin.integrate(cmodel, cq, cDq)])

# Casadi function to map joints configuration to COM position
com_position = casadi.Function("com", [cq], [cpin.centerOfMass(cmodel, cdata, cq)])

# Compute the forward kinematics and store the data in 'cdata'
# Note that now cdata is filled with symbols, so there is no need to compute the forward kinematics at every variation of q
# Since everything is a symbol, a substituition (which is what casadi functions do) is enough
cpin.framesForwardKinematics(cmodel, cdata, cq)
cpin.updateFramePlacements(cmodel, cdata)

base_rotation = casadi.Function("com", [cq], [cdata.oMf[IDX_BASE].rotation])

# Casadi functions can't output a SE3 element, so the oMf matrices are split in rotational and translational components
lf_position = casadi.Function("lf_pos", [cq], [cdata.oMf[IDX_LF].translation])
lf_rotation = casadi.Function("lf_rot", [cq], [cdata.oMf[IDX_LF].rotation])
rf_position = casadi.Function("rf_pos", [cq], [cdata.oMf[IDX_RF].translation])
rf_rotation = casadi.Function("rf_rot", [cq], [cdata.oMf[IDX_RF].rotation])

lg_position = casadi.Function("lg_pos", [cq], [cdata.oMf[IDX_LG].translation])
lg_rotation = casadi.Function("lg_rot", [cq], [cdata.oMf[IDX_LG].rotation])
le_rotation = casadi.Function("le_rot", [cq], [cdata.oMf[IDX_LE].rotation])
le_translation = casadi.Function("le_pos", [cq], [cdata.oMf[IDX_LE].translation])

rg_position = casadi.Function("rg_pos", [cq], [cdata.oMf[IDX_RG].translation])
rg_rotation = casadi.Function("rg_rot", [cq], [cdata.oMf[IDX_RG].rotation])
re_rotation = casadi.Function("re_rot", [cq], [cdata.oMf[IDX_RE].rotation])
re_translation = casadi.Function("re_pos", [cq], [cdata.oMf[IDX_RE].translation])

log = casadi.Function("log", [R, R_ref], [cpin.log3(R.T @ R_ref)])


# Defining weights
parallel_cost = 1e3
distance_cost = 1e3
straightness_body_cost = 1e3
elbow_distance_cost = 1e1
distance_btw_hands = 0.3

opti = casadi.Opti()

# Note that here the optimization variables are Dq, not q, and q is obtained by integrating.
# q = q + Dq, where the plus sign is intended as an integrator (because nq is different from nv)
# It is also possible to optimize directly in q, but in that case a constraint must be added in order to have
# the norm squared of the quaternions = 1
Dqs = opti.variable(nDq)
qs = integrate(q0, Dqs)

cost = casadi.sumsqr(qs - q0)

# Distance between the hands
cost += distance_cost * casadi.sumsqr(lg_position(qs) - rg_position(qs) - np.array([0, distance_btw_hands, 0]))

# Cost on parallelism of the two hands
r_ref = pin.utils.rotate("x", 3.14 / 2)  # orientation target
cost += parallel_cost * casadi.sumsqr(log(rg_rotation(qs), r_ref))
r_ref = pin.utils.rotate("x", -3.14 / 2)  # orientation target
cost += parallel_cost * casadi.sumsqr(log(lg_rotation(qs), r_ref))

# Body in a straight position
cost += straightness_body_cost * casadi.sumsqr(log(base_rotation(qs), base_rotation(q0)))

cost += elbow_distance_cost * casadi.sumsqr(le_translation(qs)[1] - 2) + elbow_distance_cost * casadi.sumsqr(re_translation(qs)[1] + 2)

opti.minimize(cost)

# COM
opti.subject_to(opti.bounded(-0.1, com_position(qs)[0], 0.1))
opti.subject_to(opti.bounded(-0.02, com_position(qs)[1], 0.02))
opti.subject_to(opti.bounded(0.7, com_position(qs)[2], 0.9))

# Standing foot
opti.subject_to(rf_position(qs) - rf_position(q0) == 0)
opti.subject_to(rf_rotation(qs) - rf_rotation(q0) == 0)

# Free foot
opti.subject_to(lf_position(qs)[2] >= 0.4)
opti.subject_to(opti.bounded(0.05, lf_position(qs)[0:2], 0.1))

r_ref = pin.utils.rotate("z", 3.14 / 2) @ pin.utils.rotate("y", 3.14 / 2)  # orientation target
opti.subject_to(opti.bounded(-0.0, lf_rotation(qs) - r_ref, 0.0))

# Left hand constraint to be at a certain height
opti.subject_to(opti.bounded(1.1, rg_position(qs)[2], 1.2))
opti.subject_to(opti.bounded(-distance_btw_hands / 2, rg_position(qs)[1], 0))

# Solve it !
opti.solver("ipopt")
opti.set_initial(Dqs, np.zeros(nDq))

try:
    sol = opti.solve_limited()
    # sol = opti.solve()
    # sol_xs = [opti.value(var_x) for var_x in qs]
    # sol_as = [opti.value(var_a) for var_a in Dqs]
except:
    print("ERROR in convergence, plotting debug info.")
    # sol_xs = [opti.debug.value(var_x) for var_x in qs]
    # sol_as = [opti.debug.value(var_a) for var_a in Dqs]
print("Final cost is: ", opti.value(cost))
q_sol = integrate(q0, opti.value(Dqs)).full()


def displayScene(q, dt=1e-1):
    """
    Given the robot configuration, display:
    - the robot
    - a box representing tool_id
    - a box representing in_world_M_target
    """
    pin.framesForwardKinematics(model, data, q)
    # M = data.oMf[tool_id]
    # viz.applyConfiguration(boxID, in_world_M_target)
    # viz.applyConfiguration(tipID, M)
    # for c in contacts:
    #     viz.applyConfiguration(c.viz, data.oMf[c.id])
    viz.display(q)
    time.sleep(dt)


def displayTraj(qs, dt=1e-2):
    for q in qs[1:]:
        displayScene(q, dt=dt)


while True:
    displayScene(q_sol)
    # displayTraj([x[:nq] for x in sol_xs], DT)
