import numpy as np
import pinocchio.casadi as cpin
import pinocchio as pin
import matplotlib.pyplot as plt
import sys
from ocp_utils import *
from model_legged_robot import Quadruped

nbJoint = 2
linkLength = 0.1
baseWidth = linkLength
baseLength = 2.5 * linkLength
baseHeight = linkLength / 2
floatingMass = 0.3
linkMass = 0.1
uMax = 2.5

N = 100
T = 1  # final time
dt = T / N

robot = Quadruped(nbJoint, baseWidth, baseLength, baseHeight, linkLength, floatingMass, linkMass, baseType="euler3d", RX=True)
robot.display(robot.q0)

robot.model.effortLimit = uMax * np.ones((nbJoint + robot.RX) * robot.nLegs)
robot.model.gravity.linear = np.array([0, 0, -9.81])

cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()

nq = cmodel.nq
nv = cmodel.nv
nu = (nbJoint + robot.RX) * robot.nLegs

cq = casadi.SX.sym("q", nq)
cv = casadi.SX.sym("v", nv)
phases = 2
t = casadi.SX.sym("t", phases)
ctau = casadi.SX.sym("tau", nu)
x = casadi.vertcat(cq, cv)
u = casadi.vertcat(ctau)

"""
    Dynamics
"""

# Underactuation
ctau_joints = casadi.vertcat(np.zeros(nv - nu), ctau)

# Constrained dynamics
# getting the joint acceleration
a = cpin.aba(cmodel, cdata, cq, cv, ctau_joints)
# integrator
v_next = cv + a * dt
q_next = cpin.integrate(cmodel, cq, cv * dt)
x_next = casadi.vertcat(q_next, v_next)
# state transition function Phi(x, u) -> x+
Phi = casadi.Function("Phi", [x, u], [x_next], ["x", "u"], ["x_next"])

# Constrained case
# Computing the jacobian and drift
import re

foot_regex = re.compile("foot*")
frame_names = [i.name for i in cmodel.frames]
foot_names = [name for name in frame_names if foot_regex.match(name)]
foot_frame_Ids = [cmodel.getFrameId(name) for name in foot_names]
reference_frame = cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED
drifts = []
Jacobians = []
for foot_frame_Id in foot_frame_Ids:
    J = cpin.computeFrameJacobian(cmodel, cdata, cq, foot_frame_Id, reference_frame)
    drift = frameAcceleration(cmodel, cdata, cq, cv, SX_zeros(nv), foot_frame_Id, True, reference_frame)

    # Eliminating the singular part if necessary
    if robot.RX:
        drift2d = casadi.vertcat(drift.linear)
        J2d = casadi.vertcat(J[:3, :])
    else:
        drift2d = casadi.vertcat(drift.linear[0], drift.linear[2])
        J2d = casadi.vertcat(J[0, :], J[2, :])

    drifts.append(drift2d)
    Jacobians.append(J2d)

J_contacts = casadi.vertcat(*Jacobians)
drift_contacts = casadi.vertcat(*drifts)

cpin.computeAllTerms(cmodel, cdata, cq, cv)
a_contact = cpin.forwardDynamics(cmodel, cdata, cq, cv, ctau_joints, J_contacts, drift_contacts, 1e-9)

# Euler integrator
v_next_contact = cv + a_contact * dt
q_next_contact = cpin.integrate(cmodel, cq, cv * dt)
x_next_contact = casadi.vertcat(q_next_contact, v_next_contact)
# state transition function Phi(x, u) -> x+
Phi_contact = casadi.Function("Phi_contact", [x, u], [x_next_contact], ["x", "u"], ["x_next_contact"])

"""
    Optimization problem
"""
# Casadi optimization class
opti = casadi.Opti()

# Variables MX type
X = opti.variable(x.size()[0], N + 1)  # state trajectory
U = opti.variable(u.size()[0], N)  # control trajectory

# Boundary conditions
q0 = robot.q0
# Bend the knee symmetrically
angle = np.pi / 6
q0[2] = 2 * robot.linkLength * np.cos(angle)
bent_joint = [np.pi - angle, 2 * angle]
if robot.RX:
    bent_joint = [0] + bent_joint
q0[-(nbJoint + robot.RX) * robot.nLegs :] = np.array(bent_joint * robot.nLegs)

x0 = np.hstack((q0, np.zeros(nv)))
print(f"Starting position: {x0[:nq]}")
qF = robot.q0
print(f"Target position: {qF}")

# Objective function
obj = 0
# Lagrange term
for i in range(N):
    obj += 1e-4 * X[-nv:, i].T @ X[-nv:, i] + 1e-2 * U[:, i].T @ U[:, i]

opti.minimize(obj)

# Dynamic constraints
for k in range(int(N)):
    if k < int(N / 2):
        opti.subject_to(X[:, k + 1] == Phi_contact(X[:, k], U[:, k]))
        # opti.subject_to(X[3:6, :] == robot.q0[3:6])
    else:
        # opti.subject_to(X[:, k + 1] == Phi_contact(X[:,k], U[:,k]))
        opti.subject_to(X[:, k + 1] == Phi(X[:, k], U[:, k]))

# Path constraints
# control limits, here the numerical values must be used
opti.subject_to(opti.bounded(-robot.model.effortLimit, U, robot.model.effortLimit))

# Hard constraints
opti.subject_to(X[:, 0] == x0)

# Lateral jump
opti.subject_to(X[2:nq, -1] == qF[2:nq])
opti.subject_to(X[:2, -1] == np.array([0, robot.linkLength]))

# Miscellaneous constraints
# ground bounds
opti.subject_to(X[2, :] >= robot.linkLength * np.sqrt(angle) / 2)

# Imposing the joint constraints
lower_joint_regex = re.compile(f"^leg_.*_RY_{robot.nbJoint}$")
lower_joint_names = [name for name in frame_names if lower_joint_regex.match(name)]
lower_joint_frames_Ids = [cmodel.getFrameId(name) for name in lower_joint_names]
for frameJoint in lower_joint_frames_Ids:
    R, pos = framePlacementFunctions(cmodel, cdata, cq, cv, frameJoint)
    for i in range(X.shape[1]):
        opti.subject_to(pos(X[2, i]) >= robot.linkLength / 10)

# Initial values for solver
opti.set_initial(X, np.vstack([x0 for _ in range(N + 1)]).T)

"""
    Solver
"""

# Options
opts = {"verbose": False}
opts["ipopt"] = {"max_iter": 1000, "linear_solver": "mumps"}

# Initialization
opti.solver("ipopt", opts)  # set numerical backend

"""
    Showing the solution or the unconverged last step
"""


def plot_solution(sol, T, N):
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, constrained_layout=True)

    dt = T / float(N)
    time = np.arange(0, T + dt, dt)

    [ax0.plot(time, sol.value(X[i, :])) for i in range(nq)]
    ax0.legend([f"$q_{{{i}}}$" for i in range(nq)])
    ax0.set_xlabel("t [s]")
    ax0.set_title("x")

    [ax1.plot(time, sol.value(X[i, :])) for i in range(nq, nq + nv)]
    ax1.legend([f"$v_{{{i}}}$" for i in range(nv)])
    ax1.set_xlabel("t [s]")
    ax1.set_title("v")

    [ax2.plot(time[:-1], sol.value(U[i, :])) for i in range(nu)]
    ax2.legend([f"$u_{{{i}}}$" for i in range(nu)])
    ax2.set_xlabel("t [s]")
    ax2.set_title("u")

    plt.show()


try:
    # launch the solver
    import time

    times = []
    for i in range(1):
        tic = time.time()
        sol = opti.solve_limited()
        toc = time.time()
        times.append(toc - tic)

    print(f"Mean {np.mean(times)}, sd {np.std(times)}")

    # if converged use plot the converged solution otherwise the last step in exception
    plot_solution(sol, T, N)

except:
    import warnings

    sol = opti.debug
    plot_solution(sol, T, N)
    warnings.simplefilter("error", UserWarning)
    warnings.warn("Problem NOT converged, showing just for debug")

"""
    Simulation of the solution in GV
"""
import time

scaling_factor = 1

for i in range(10):
    q = sol.value(X[: cmodel.nq, :])
    q_traj = np.vstack(q)
    for i in range(q_traj.shape[1]):
        robot.display(q_traj[:, i])
        time.sleep(dt * scaling_factor)
    time.sleep(0.5)
