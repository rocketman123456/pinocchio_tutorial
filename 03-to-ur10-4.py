import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
from meshcat_viewer_wrapper import MeshcatVisualizer
from types import SimpleNamespace
import time

# robot arm example with collision

# --- Load robot model
robot = robex.load("ur10")
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)

geom = robot.collision_model.geometryObjects[0]
shape = geom.geometry
print(shape)
vertices = shape.vertices()
print(vertices)

# cw = casadi.SX.sym("w", 3)
# exp = casadi.Function("exp3", [cw], [cpin.exp3(cw)])

# opti = casadi.Opti()
# var_w = opti.variable(3)
# var_r = opti.variable(3)
# var_c = opti.variable(3)

# # The ellipsoid matrix is represented by w=log3(R),diag(P) with R,P=eig(A)
# R = exp(var_w)
# A = R @ casadi.diag(1 / var_r**2) @ R.T

# totalcost = var_r[0] * var_r[1] * var_r[2]

# opti.subject_to(var_r >= 0)

# for g_v in vertices:
#     # g_v is the vertex v expressed in the geometry frame.
#     # Convert point from geometry frame to joint frame
#     j_v = geom.placement.act(g_v)
#     # Constraint the ellipsoid to be including the point
#     opti.subject_to((j_v - var_c).T @ A @ (j_v - var_c) <= 1)

# opti.minimize(totalcost)
# opti.solver("ipopt")  # set numerical backend
# opti.set_initial(var_r, 10)

# sol = opti.solve_limited()

# sol_r = opti.value(var_r)
# sol_A = opti.value(A)
# sol_c = opti.value(var_c)
# sol_R = opti.value(exp(var_w))

# Build the ellipsoid 3d shape
# Ellipsoid in meshcat
# viz.addEllipsoid("el", sol_r, [0.3, 0.9, 0.3, 0.3])
# # jMel is the placement of the ellipsoid in the joint frame
# jMel = pin.SE3(sol_R, sol_c)

# Place the body, the vertices and the ellispod at a random configuration oMj_rand
# oMj_rand = pin.SE3.Random()
# viz.applyConfiguration(viz.getViewerNodeName(geom, pin.VISUAL), oMj_rand)
# for i in np.arange(0, vertices.shape[0]):
#     viz.applyConfiguration(f"world/point_{i}", oMj_rand.act(vertices[i]).tolist() + [1, 0, 0, 0])
# viz.applyConfiguration("el", oMj_rand * jMel)

### HYPER PARAMETERS
Mtarget = pin.SE3(pin.utils.rotate("y", 3), np.array([-0.8, -0.1, 0.2]))  # x,y,z
q0 = np.array([0, 5, 3, 0, 2, 0])
endEffectorFrameName = "tool0"

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
endEffector_ID = model.getFrameId(endEffectorFrameName)

# These values are computed using encapsulating_ellipse.py
ellipses = [
    SimpleNamespace(
        name="shoulder_lift_joint",
        A=np.array(
            [
                [75.09157846, 0.34008563, -0.08817025],
                [0.34008563, 60.94969446, -0.55672959],
                [-0.08817025, -0.55672959, 3.54456814],
            ]
        ),
        center=np.array([-1.05980885e-04, -5.23471160e-02, 2.26280651e-01]),
    ),
    SimpleNamespace(
        name="elbow_joint",
        A=np.array(
            [
                [1.30344372e02, -5.60880392e-02, -1.87555288e-02],
                [-5.60880392e-02, 9.06119040e01, 1.65531606e-01],
                [-1.87555288e-02, 1.65531606e-01, 4.08568387e00],
            ]
        ),
        center=np.array([-2.01944435e-05, 7.22262249e-03, 2.38805264e-01]),
    ),
    SimpleNamespace(
        name="wrist_1_joint",
        A=np.array(
            [
                [2.31625634e02, 5.29558437e-01, -1.62729657e-01],
                [5.29558437e-01, 2.18145143e02, -1.42425434e01],
                [-1.62729657e-01, -1.42425434e01, 1.73855962e02],
            ]
        ),
        center=np.array([-9.78431524e-05, 1.10181763e-01, 6.67932259e-03]),
    ),
    SimpleNamespace(
        name="wrist_2_joint",
        A=np.array(
            [
                [2.32274519e02, 1.10812959e-01, -1.12998357e-02],
                [1.10812959e-01, 1.72324444e02, -1.40077876e01],
                [-1.12998357e-02, -1.40077876e01, 2.19132854e02],
            ]
        ),
        center=np.array([-2.64650554e-06, 6.27960760e-03, 1.11112087e-01]),
    ),
]

for e in ellipses:
    e.id = robot.model.getJointId(e.name)
    l, P = np.linalg.eig(e.A)
    e.radius = 1 / l**0.5
    e.rotation = P
    e.placement = pin.SE3(P, e.center)

# Obstacle positions are arbitrary. Their radius is meaningless, just for visualization.
obstacles = [SimpleNamespace(radius=0.01, pos=np.array([-0.4, 0.2 + s, 0.5]), name=f"obs_{i_s}") for i_s, s in enumerate(np.arange(-0.5, 0.5, 0.1))]

for e in ellipses:
    viz.addEllipsoid(f"el_{e.name}", e.radius, [0.3, 0.9, 0.3, 0.3])
for io, o in enumerate(obstacles):
    viz.addSphere(f"obs_{io}", o.radius, [0.8, 0.3, 0.3, 0.9])

# --- Add box to represent target
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
    - a box representing endEffector_ID
    - a box representing Mtarget
    """
    pin.framesForwardKinematics(model, data, q)
    M = data.oMf[endEffector_ID]
    viz.applyConfiguration(boxID, Mtarget)
    viz.applyConfiguration(tipID, M)
    for e in ellipses:
        M = data.oMi[e.id]
        viz.applyConfiguration(f"el_{e.name}", M * e.placement)
    for io, o in enumerate(obstacles):
        viz.applyConfiguration(f"obs_{io}", pin.SE3(np.eye(3), o.pos))
    viz.display(q)
    time.sleep(dt)


# --- Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()

cq = casadi.SX.sym("q", model.nq, 1)
cpin.framesForwardKinematics(cmodel, cdata, cq)
error6_tool = casadi.Function(
    "etool",
    [cq],
    [cpin.log6(cdata.oMf[endEffector_ID].inverse() * cpin.SE3(Mtarget)).vector],
)
error3_tool = casadi.Function(
    "etool",
    [cq],
    [cdata.oMf[endEffector_ID].translation - Mtarget.translation],
)
error_tool = error3_tool

cpos = casadi.SX.sym("p", 3)
for e in ellipses:
    # Position of the obstacle cpos in the ellipse frame.
    e.e_pos = casadi.Function(
        f"e{e.id}",
        [cq, cpos],
        [cdata.oMi[e.id].inverse().act(casadi.SX(cpos))],
    )

opti = casadi.Opti()
var_q = opti.variable(model.nq)
totalcost = casadi.sumsqr(error_tool(var_q))

for e in ellipses:
    for o in obstacles:
        # obstacle position in ellipsoid (joint) frame
        e_pos = e.e_pos(var_q, o.pos)
        opti.subject_to((e_pos - e.center).T @ e.A @ (e_pos - e.center) >= 1)

opti.minimize(totalcost)
p_opts = dict(print_time=False, verbose=False)
s_opts = dict(print_level=0)
opti.solver("ipopt")  # set numerical backend
opti.set_initial(var_q, robot.q0)

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_q = opti.value(var_q)
except:
    print("ERROR in convergence, plotting debug info.")
    sol_q = opti.debug.value(var_q)

while True:
    displayScene(robot.q0, 1)
    displayScene(sol_q, 1)
