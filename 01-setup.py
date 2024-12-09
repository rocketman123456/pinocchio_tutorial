import meshcat
import example_robot_data
import pinocchio
import time
import numpy as np
import casadi
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

# load robot model
talos = example_robot_data.load("talos")

viz = MeshcatVisualizer(talos.model, talos.collision_model, talos.visual_model)
viz.initViewer(loadModel=True, open=True)
# Load the robot in the viewer.
viz.loadViewerModel()
viz.displayVisuals(True)
# viz.viewer.jupyter_cell()

# Define a robot configuration (joint positions)
# q = pin.neutral(talos)  # Neutral configuration (all joints at zero)
q0 = talos.q0
v0 = talos.v0

differentiable_talos = cpin.Model(talos.model)
differentiable_data = differentiable_talos.createData()
cpin.forwardKinematics(differentiable_talos, differentiable_data, casadi.SX(q0))

curr_time = 0.0

while True:
    # Generate a simple animation
    for i in range(100):
        q = talos.q0  # pin.neutral(talos)
        q[1] = np.sin(curr_time)  # Example: oscillate a joint
        # curr_time += 0.05
        viz.display(q)
        time.sleep(0.05)  # Pause to simulate real-time
    viz.display(q0)
