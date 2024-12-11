import example_robot_data as robex
import numpy as np
import pinocchio
import crocoddyl
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

# ### Load robot
robot = robex.load("talos")
robot_model = robot.model
# The robot data will be used at config time to define some values of the OCP
robot_data = robot_model.createData()

lims = robot_model.effortLimit
# lims[19:] *= 0.5  # reduced artificially the torque limits
robot_model.effortLimit = lims

# crocoddyl state action
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# ### Hyperparameters

# Set integration time
DT = 5e-2
T = 40

# Initialize reference state, target and reference CoM

base_frameName = "base_link"
hand_frameName = "gripper_left_joint"
rightFoot_frameName = "right_sole_link"
leftFoot_frameName = "left_sole_link"

# Main frame Ids
hand_id = robot_model.getFrameId(hand_frameName)
rightFoot_id = robot_model.getFrameId(rightFoot_frameName)
leftFoot_id = robot_model.getFrameId(leftFoot_frameName)
base_id = robot_model.getFrameId(base_frameName)

# Init state
q0 = robot_model.referenceConfigurations["half_sitting"]
x0 = np.concatenate([q0, np.zeros(robot_model.nv)])

# Reference quantities
pinocchio.framesForwardKinematics(robot_model, robot_data, q0)
pinocchio.updateFramePlacements(robot_model, robot_data)

comRef = (robot_data.oMf[rightFoot_id].translation + robot_data.oMf[leftFoot_id].translation) / 2
comRef[2] = 0.8 * pinocchio.centerOfMass(robot_model, robot_data, q0)[2].item()
rfPos0 = robot_data.oMf[rightFoot_id].translation
lfPos0 = robot_data.oMf[leftFoot_id].translation
handPos0 = robot_data.oMf[hand_id].translation
basePos = robot_data.oMf[base_id].translation

print(f"comRef : {comRef}")

in_world_M_base_target = pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 1.0]))
in_world_M_foot_target_1 = pinocchio.SE3(np.eye(3), np.array([0.0, 0.4, 0.1]))
in_world_M_foot_target_2 = pinocchio.SE3(np.eye(3), np.array([0.3, 0.15, 0.35]))
in_world_M_foot_target_3 = pinocchio.SE3(np.eye(3), np.array([0.3, 0.15, 0.0]))
in_world_M_hand_target = pinocchio.SE3(np.eye(3), np.array([0.4, 0, 1.2]))
# in_world_M_foot_target_1 = pinocchio.SE3(pinocchio.utils.rpyToMatrix(0, np.pi / 4, 0), np.array([0.0, 0.1, 0.0]))
# in_world_M_foot_target_2 = pinocchio.SE3(np.eye(3), np.array([0.3, 0.3, 0.35]))
# in_world_M_hand_target = pinocchio.SE3(np.eye(3), np.array([0.4, 0, 1.2]))

# ### Contact model
# Create two contact models used along the motion
supportContactModelLeft = crocoddyl.ContactModel6D(
    state,
    leftFoot_id,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL,
    actuation.nu,
    np.array([0, 40]),
)
supportContactModelRight = crocoddyl.ContactModel6D(
    state,
    rightFoot_id,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL,
    actuation.nu,
    np.array([0, 40]),
)

contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel1Foot.addContact(rightFoot_frameName + "_contact", supportContactModelRight)

contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel2Feet.addContact(leftFoot_frameName + "_contact", supportContactModelLeft)
contactModel2Feet.addContact(rightFoot_frameName + "_contact", supportContactModelRight)

# ### Cost model
# Cost for joint limits
maxfloat = 1e25
# print(f"lower bound: {robot_model.lowerPositionLimit[7:]}")
# print(f"upper bound: {robot_model.upperPositionLimit[7:]}")
xlb = np.concatenate(
    [
        -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        robot_model.lowerPositionLimit[7:],
        -maxfloat * np.ones(state.nv),
    ]
)
xub = np.concatenate(
    [
        maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        robot_model.upperPositionLimit[7:],
        maxfloat * np.ones(state.nv),
    ]
)
bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

# Cost for state and control
xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv) ** 2)
uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
xTActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv) ** 2)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

# Cost for target reaching: hand and foot
handTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, hand_id, in_world_M_hand_target, actuation.nu)
handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1] * 3 + [0.0001] * 3) ** 2)
handTrackingCost = crocoddyl.CostModelResidual(state, handTrackingActivation, handTrackingResidual)

# For the flying foot, we define two targets to successively reach
footTrackingResidual1 = crocoddyl.ResidualModelFramePlacement(state, leftFoot_id, in_world_M_foot_target_1, actuation.nu)
footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1, 1, 0.1] + [1.0] * 3) ** 2)
footTrackingCost1 = crocoddyl.CostModelResidual(state, footTrackingActivation, footTrackingResidual1)
footTrackingResidual2 = crocoddyl.ResidualModelFramePlacement(state, leftFoot_id, in_world_M_foot_target_2, actuation.nu)
footTrackingCost2 = crocoddyl.CostModelResidual(state, footTrackingActivation, footTrackingResidual2)
footTrackingResidual3 = crocoddyl.ResidualModelFramePlacement(state, leftFoot_id, in_world_M_foot_target_3, actuation.nu)
footTrackingCost3 = crocoddyl.CostModelResidual(state, footTrackingActivation, footTrackingResidual3)

# Cost for CoM reference
comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
comTrack = crocoddyl.CostModelResidual(state, comResidual)

# Cost for base
baseTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, base_id, in_world_M_base_target, actuation.nu)
baseTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1] * 3 + [0.0001] * 3) ** 2)
baseTrackingCost = crocoddyl.CostModelResidual(state, baseTrackingResidual)

# Create cost model per each action model. We divide the motion in 4 phases plus its terminal model.

# Phase 1: two feet in contact, hand reach the target
runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel1.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel1.addCost("basePose", baseTrackingCost, 1e2)
runningCostModel1.addCost("limitCost", limitCost, 1e3)
dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel2Feet, runningCostModel1)
runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)

# Phase 2: only right foot in contact, hand stays on target, left foot to target 1
runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel2.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel2.addCost("footPose", footTrackingCost1, 1e2)
runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel2.addCost("limitCost", limitCost, 1e3)
runningCostModel2.addCost("basePose", baseTrackingCost, 1e2)
dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot, runningCostModel2)
runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)

# Phase 3: only right foot in contact, hand stays on target, left foot to target 2
runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel3.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel3.addCost("footPose", footTrackingCost2, 1e2)
runningCostModel3.addCost("stateReg", xRegCost, 1e-3)
runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel3.addCost("limitCost", limitCost, 1e3)
runningCostModel3.addCost("basePose", baseTrackingCost, 1e2)
dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot, runningCostModel3)
runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)

# Phase 4: two feet in contact, hand reach the target
runningCostModel4 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel4.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel4.addCost("stateReg", xRegCost, 1e-3)
runningCostModel4.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel4.addCost("limitCost", limitCost, 1e3)
runningCostModel4.addCost("basePose", baseTrackingCost, 1e2)
dmodelRunning4 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel2Feet, runningCostModel4)
runningModel4 = crocoddyl.IntegratedActionModelEuler(dmodelRunning4, DT)

# Terminal cost: nothing specific
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel.addCost("gripperPose", handTrackingCost, 1e2)
terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
# terminalCostModel.addCost("comPose", comTrack, 1e1)
terminalCostModel.addCost("basePose", baseTrackingCost, 1e2)
terminalCostModel.addCost("limitCost", limitCost, 1e3)
dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot, terminalCostModel)
terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

problem = crocoddyl.ShootingProblem(x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T + [runningModel4] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverFDDP(problem)
solver.th_stop = 1e-7
solver.setCallbacks(
    [
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackLogger(),
    ]
)

# ### Warm start from quasistatic solutions
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(xs, us, 500, False, 1e-9)

# ### DISPLAY
# Initialize viewer
from meshcat_viewer_wrapper import MeshcatVisualizer

viz = MeshcatVisualizer(robot, url=None)
viz.display(robot.q0)

while True:
    viz.addBox("world/hand", [0.1, 0.1, 0.1], "blue")
    viz.applyConfiguration("world/hand", in_world_M_hand_target)

    viz.addBox("world/foot1", [0.1, 0.1, 0.1], [0, 1, 0, 1])
    viz.applyConfiguration("world/foot1", in_world_M_foot_target_1)

    viz.addBox("world/foot2", [0.1, 0.1, 0.1], "red")
    viz.applyConfiguration("world/foot2", in_world_M_foot_target_2)

    viz.play([x[: robot.model.nq] for x in solver.xs], DT)
