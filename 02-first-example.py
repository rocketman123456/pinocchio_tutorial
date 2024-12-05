import time
import unittest
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
# viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True, open=True)
viz.display(robot.q0)

robot.q0 = np.array([0, -np.pi / 2, 0, 0, 0, 0])

tool_id = model.getFrameId("tool0")

in_world_M_target = pin.SE3(
    pin.utils.rotate("x", np.pi / 4),
    np.array([-0.5, 0.1, 0.2]),
)

# --- Add box to represent target

# Add a vizualization for the target
boxID = "world/box"
# material = materialFromColor([1.0, 0.2, 0.2, 0.5])
# viz.viewer[boxID].set_object(meshcat.geometry.Box([0.05, 0.1, 0.2]), material)
viz.addBox(boxID, [0.05, 0.1, 0.2], [1.0, 0.2, 0.2, 0.5])

# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [0.08] * 3, [0.2, 0.2, 1.0, 0.5])
# material = materialFromColor([0.2, 0.2, 1.0, 0.5])
# viz.viewer[tipID].set_object(meshcat.geometry.Box([0.08] * 3), material)

viz.applyConfiguration(tipID, in_world_M_target)
# R, p = in_world_M_target.rotation, in_world_M_target.translation
# T = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
# viz.viewer[tipID].set_transform(T)

# --- Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()
cq = casadi.SX.sym("q", model.nq, 1)
print(3 * (cq + 1))
cpin.framesForwardKinematics(cmodel, cdata, cq)
print(cdata.oMf[tool_id])


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


while True:
    viz.display(robot.q0)
