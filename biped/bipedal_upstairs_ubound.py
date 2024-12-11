import os
import pickle
import signal
import sys
from pathlib import Path

import example_robot_data
import numpy as np
import pinocchio as pnc
import toml

# from biped_convert import build_graph_from_trajectories, build_scenario, get_trajectories_list
from biped_plot import plotSolution
from biped_upstairs_from_dict import SimpleBipedGaitProblem as Upstairs

import crocoddyl
import mim_solvers

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
CONVERT = "conv" in sys.argv or "CROCODDYL_CONV" in os.environ
GEN_GUESS = "gen_guess" in sys.argv
NO_IMPULSE = "no_impulse" in sys.argv

if "V2" in sys.argv:
    VERSION = "V2"
elif "V3" in sys.argv:
    VERSION = "V3"
elif "V4" in sys.argv:
    VERSION = "V4"
elif "V5" in sys.argv:
    VERSION = "V5"
elif "lowcost" in sys.argv:
    VERSION = "lowcost"
else:
    VERSION = False

FLYING_SIDE = "right" if "right" in sys.argv else "left"
signal.signal(signal.SIGINT, signal.SIG_DFL)
EVE = False

stepLength = 0.39  # 0.6
stepHeight = 0.1  # 0.1
extraStepHeight = 0.05
toeClearance = 0.06
guessHeight = 0.17  # 0.1
stairsHeight = 0.1  # 0.1
initial_guess_path = Path(f"./initial_guess_{int(1e2*guessHeight)}cm.npz")

# Load model
rightToe = None
leftToe = None
rightHeel = None
leftHeel = None
leftFootPolygon = None
rightFootPolygon = None
if EVE:
    from wdc.walkgen_py.helpers import EveBetaHelper
    from wdc.walkgen_py.json_generator import JsonGenerator

    urdf_path = Path.home() / "wdc_workspace/src/wandercode/wanderbrain/products/eve_beta/data/trajectories/exo_with_patient.urdf"
    eve = pnc.RobotWrapper()
    eve.initFromURDF(str(urdf_path), root_joint=pnc.JointModelFreeFlyer())
    robot = eve
    generator = JsonGenerator(EveBetaHelper())
    standing_pose_path = Path.home() / "wdc_workspace/src/wandercode/wanderbrain/products/eve_beta/data/trajectories/standing_pose.traj"
    standing_pose = generator.read_json_posture(standing_pose_path)
    q0 = standing_pose
    rightFoot = "RightSole"
    leftFoot = "LeftSole"
    rightToe = "RightMetatarsusProjected"
    leftToe = "LeftMetatarsusProjected"
    rightHeel = "RightHeelProjected"
    leftHeel = "LeftHeelProjected"
    leftFootPolygon = [
        "LeftInternalToeLimit",
        "LeftInternalMetatarsusLimit",
        "LeftExternalMetatarsusLimit",
        "LeftExternalToeLimit",
        "LeftExternalHeelLimit",
        "LeftInternalHeelLimit",
    ]
    rightFootPolygon = [
        "RightExternalToeLimit",
        "RightExternalMetatarsusLimit",
        "RightInternalMetatarsusLimit",
        "RightInternalToeLimit",
        "RightInternalHeelLimit",
        "RightExternalHeelLimit",
    ]
    talos_legs = example_robot_data.load("talos_legs")
else:
    talos_legs = example_robot_data.load("talos_legs")
    lims = talos_legs.model.effortLimit
    # lims *= 0.5  # reduced artificially the torque limits
    talos_legs.model.lowerPositionLimit[-2:] /= 2
    talos_legs.model.lowerPositionLimit[-2 - 6 : -2] /= 2
    talos_legs.model.effortLimit = lims
    robot = talos_legs
    q0 = robot.model.referenceConfigurations["half_sitting"].copy()
    rightFoot = "right_sole_link"
    leftFoot = "left_sole_link"

# Defining the initial state of the robot
v0 = pnc.utils.zero(robot.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
options = toml.load(f"config/config_upstairs_{str(VERSION)}.toml")
gait = Upstairs(
    robot.model,
    q0,
    FLYING_SIDE,
    rightFoot,
    leftFoot,
    options=options,
    leftToe=leftToe,
    rightToe=rightToe,
    integrator="euler",
    control="zero",
    no_impulse=NO_IMPULSE,
)

# Setting up all tasks
PARAMS = {
    "stepLength": stepLength,
    "stepHeight": stepHeight,
    "extraStepHeight": extraStepHeight,
    "stairsHeight": guessHeight if GEN_GUESS else stairsHeight,
    "toeClearance": toeClearance,
    "timeStep": 0.03,
    "stepKnots": 50,
    "supportKnots": 25,
}
cameraTF = [3.0, 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]


# Creating a walking problem
problem, feetTargets, comTargets = gait.createUpstairsProblem(
    x0,
    PARAMS["stepLength"],
    PARAMS["stepHeight"],
    PARAMS["extraStepHeight"],
    PARAMS["stairsHeight"],
    PARAMS["toeClearance"],
    PARAMS["timeStep"],
    PARAMS["stepKnots"],
    PARAMS["supportKnots"],
)
print(f'Using hard constraints: {options["constraint"]["hard"]["use"]}')
if not options["constraint"]["hard"]["use"]:
    solver = crocoddyl.SolverBoxFDDP(problem)
    solver.th_stop = 1e-9
else:
    solver = mim_solvers.SolverCSQP(problem)
    solver.termination_tolerance = 1e-2  # Termination criteria (KKT residual)
    solver.max_qp_iters = 1000  # Maximum number of QP iteration
    solver.eps_abs = 1e-5  # QP termination absolute criteria, 1e-9
    solver.eps_rel = 0.0  # QP termination absolute criteria
    solver.use_filter_line_search = True  # True by default, False = use merit function
    solver.with_callbacks = True

from repr_ocp import reprProblem

reprFilename = f"/tmp/bipedal_upstars-{VERSION}-repr.ascii"
with open(reprFilename, "w") as f:
    f.write(reprProblem(solver.problem))
    print(f"OCP described in {reprFilename}")


# Added the callback functions
print("*** SOLVE UPSTAIRS ***")
if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.getCallbacks()[0].precision = 3
solver.getCallbacks()[0].level = crocoddyl.VerboseLevel._2

# Solving the problem with the DDP solver
if initial_guess_path.exists() and not GEN_GUESS:
    with open(initial_guess_path, "rb") as inp:
        d = pickle.load(inp)
        xs = d["xs"]
        us = d["us"]
else:
    print(f"[WARNING] Initial guess file {initial_guess_path.name} not found. Trying without initial guess. (Use gen_guess to generate it)")
    xs = [x0] * (solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0] * solver.problem.T)

if not solver.solve(xs, us, 100, False, 0.1):
    print("[WARNING] SOLVER DID NOT CONVERGE.")

# Save guess
if GEN_GUESS:
    with open(initial_guess_path, "wb") as outp:
        pickle.dump({"xs": solver.xs, "us": solver.us}, outp, pickle.HIGHEST_PROTOCOL)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.MeshcatDisplay(robot, frameNames=[rightFoot, leftFoot])
    display.rate = -1
    display.freq = 1

    def launch_display(display, solver):
        while True:
            display.displayFromSolver(solver)

    from threading import Thread

    Thread(target=launch_display, args=(display, solver)).start()

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.grads,
        log.stops,
        log.steps,
        figTitle="Upstairs",
        figIndex=0,
        show=True,
    )

    plotSolution(
        solver,
        bounds=True,
        show=True,
        leftFoot=leftFoot,
        rightFoot=rightFoot,
        comTargets=comTargets,
        frameTargets=feetTargets,
        frames=[leftToe, rightToe, leftHeel, rightHeel] if EVE else [leftFoot, rightFoot],
        clearanceFrames={"left": [leftHeel, leftToe], "right": [rightHeel, rightToe]} if EVE else None,
        leftFootPolygon=leftFootPolygon if EVE else None,
        rightFootPolygon=rightFootPolygon if EVE else None,
    )

if CONVERT:
    step = lambda prefix, toe_push: [f"{prefix}_dbsp", f"{prefix}_dbsp_toe_push", f"{prefix}_sgsp"] if toe_push else [f"{prefix}_dbsp", f"{prefix}_sgsp"]

    domains = []
    for prefix in ["first", "second", "last"]:
        domains += step(prefix, TOE_PUSH and prefix == "second")
        if not NO_IMPULSE:
            domains += [
                f"{prefix}_sgsp_impulse",
            ]

    domains += [
        "final_dbsp",
    ]

    trajectories = get_trajectories_list(
        solver,
        domains,
    )

    graph = build_graph_from_trajectories(trajectories)

    scenario = build_scenario(graph, trajectories)

    scenario.to_json_file(f"upstairs_{FLYING_SIDE}_crocoddyl" + scenario.canonical_filename())
