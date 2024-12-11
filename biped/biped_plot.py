import numpy as np
import pinocchio

import crocoddyl
from tabbed_figure import TabbedFigure


def se3_to_xyzrpy(se3_transform):
    xyz = se3_transform.translation
    rpy = pinocchio.utils.matrixToRpy(pinocchio.Quaternion(se3_transform.rotation).matrix())
    return np.concatenate((xyz, rpy))


def plotSolution(
    solver,
    leftFoot,
    rightFoot,
    rmodel=None,
    bounds=True,
    figIndex=1,
    figTitle="",
    show=True,
    comTargets=None,
    frameTargets=None,
    frames=None,
    clearanceFrames=None,
    leftFootPolygon=None,
    rightFootPolygon=None,
):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    qt_app = TabbedFigure.get_qt_app()
    window = TabbedFigure()
    window.set_window_title("Trajectory Viewer (Crocoddyl)")

    leftContactNames = [leftFoot]
    rightContactNames = [rightFoot]
    for frame in frames:
        if frame.startswith("Left"):
            leftContactNames.append(frame)
        elif frame.startswith("Right"):
            rightContactNames.append(frame)
    if leftFootPolygon is not None:
        leftContactNames += leftFootPolygon
    if rightFootPolygon is not None:
        rightContactNames += rightFootPolygon

    (
        t,
        n_contacts,
        left_contact,
        right_contact,
        xs,
        us,
        left_forces,
        right_forces,
        contact_frames,
    ) = ([], [], [], [], [], [], {}, {}, set())
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub, xs_mid = [], [], []

    def fill_forces(contacts, contact_sufixes, contactNames, contactDicts, contactLists):
        for contact_names, contact_dict, contact_list in zip(contactNames, contactDicts, contactLists):
            contact_list.append(False)
            for name in contact_names:
                if name not in contact_dict.keys():
                    contact_dict[name] = []
                for suffix in contact_sufixes:
                    if (contact_name := name + suffix) in contacts.keys():
                        contact_frames.add(name)
                        contact_dict[name].append(contacts[contact_name].jMf.actInv(contacts[contact_name].f).vector)
                        contact_list[-1] = True
                        break
                else:
                    contact_dict[name].append(np.zeros(6))

    def updateTrajectories(solver):
        xs.extend(solver.xs)
        for m, d in zip(solver.problem.runningModels, solver.problem.runningDatas):
            if hasattr(m, "dt"):
                t.append(t[-1] + m.dt)
            else:
                t.append(t[-1])
            if hasattr(m, "differential"):
                if isinstance(d.differential, crocoddyl.StdVec_DiffActionData):
                    raise ValueError("[ERROR][PLOT] Does not handle StdVec_DiffActionData yet (control paramtrization above zero).")
                else:
                    us.append(d.differential.multibody.joint.tau)
                    contacts = d.differential.multibody.contacts.contacts.todict()
                n_contacts.append(len(contacts))
                fill_forces(
                    contacts,
                    ["_contact", "_contact6D", "_contact3D", "_contact5D"],
                    [leftContactNames, rightContactNames],
                    [left_forces, right_forces],
                    [left_contact, right_contact],
                )
                if bounds and isinstance(m.differential, crocoddyl.DifferentialActionModelContactFwdDynamics):
                    us_lb.extend([m.u_lb])
                    us_ub.extend([m.u_ub])
            else:
                impulses = d.multibody.impulses.impulses.todict()
                n_contacts.append(len(impulses))
                fill_forces(
                    impulses,
                    ["_impulse"],
                    [leftContactNames, rightContactNames],
                    [left_forces, right_forces],
                    [left_contact, right_contact],
                )
            if bounds:
                xs_lb.extend([m.state.lb])
                xs_ub.extend([m.state.ub])
                xs_mid.extend([(m.state.ub + m.state.lb) / 2])

    t.append(0.0)
    n_contacts.append(2)
    # xs.append(solver[0].problem.x0)
    if isinstance(solver, list):
        for s in solver:
            if rmodel is None:
                rmodel = s.problem.runningModels[0].state.pinocchio
            nq, nv, nu = (
                rmodel.nq,
                rmodel.nv,
                s.problem.runningModels[0].differential.actuation.nu,
            )
            updateTrajectories(s)
    else:
        if rmodel is None:
            rmodel = solver.problem.runningModels[0].state.pinocchio
        nq, nv, nu = (
            rmodel.nq,
            rmodel.nv,
            solver.problem.runningModels[0].differential.actuation.nu,
        )
        updateTrajectories(solver)

    # Getting the state and control trajectories
    nx = nq + nv
    X = [0.0] * nx
    U = [0.0] * nu
    if bounds:
        U_LB = [0.0] * nu
        U_UB = [0.0] * nu
        X_LB = [0.0] * nx
        X_UB = [0.0] * nx
        X_MID = [0.0] * nx
    for i in range(nx):
        X[i] = [x[i] for x in xs]
        if bounds:
            X_LB[i] = [x[i] for x in xs_lb]
            X_UB[i] = [x[i] for x in xs_ub]
            X_MID[i] = [x[i] for x in xs_mid]
    for i in range(nu):
        U[i] = [u[i] for u in us]
        if bounds:
            U_LB[i] = [u[i] for u in us_lb]
            U_UB[i] = [u[i] for u in us_ub]

    # Get CoM / CoP / transform trajectories
    rdata = rmodel.createData()
    lfId = rmodel.getFrameId(leftFoot)
    rfId = rmodel.getFrameId(rightFoot)
    CoM = np.zeros((3, len(xs)))
    dCoM = np.zeros((3, len(xs)))
    DCM = np.zeros((3, len(xs)))
    CoP = np.zeros((3, len(xs)))
    LF = np.zeros((6, len(xs)))
    RF = np.zeros((6, len(xs)))
    LFvel = np.zeros((6, len(xs)))
    RFvel = np.zeros((6, len(xs)))
    frameTransforms = {name: np.zeros((6, len(xs))) for name in frames}
    LFPolygon = np.zeros((len(leftFootPolygon) if leftFootPolygon is not None else 0, 6, len(xs)))
    RFPolygon = np.zeros((len(rightFootPolygon) if rightFootPolygon is not None else 0, 6, len(xs)))
    for i, x in enumerate(xs):
        q = x[: rmodel.nq]
        dq = x[rmodel.nq : rmodel.nq + rmodel.nv]
        pinocchio.forwardKinematics(rmodel, rdata, q, dq)
        pinocchio.updateFramePlacements(rmodel, rdata)
        pinocchio.centerOfMass(rmodel, rdata, q, dq)
        CoM[:, i] = rdata.com[0]
        dCoM[:, i] = rdata.vcom[0]
        left_foot = rdata.oMf[lfId]
        right_foot = rdata.oMf[rfId]
        left_foot_vel = pinocchio.getFrameVelocity(rmodel, rdata, lfId, pinocchio.WORLD)
        right_foot_vel = pinocchio.getFrameVelocity(rmodel, rdata, rfId, pinocchio.WORLD)

        if i < len(xs) - 1:
            left_wrench = pinocchio.Force(np.zeros(6))
            right_wrench = pinocchio.Force(np.zeros(6))
            for wrench, forces in zip([left_wrench, right_wrench], [left_forces, right_forces]):
                for name in forces.keys():
                    id = rmodel.getBodyId(name)
                    local_wrench = pinocchio.Force(forces[name][i][:])
                    wrench += rdata.oMf[id].act(local_wrench)
            useLeft = float(np.linalg.norm(left_wrench.linear) > 10.0)
            useRight = float(np.linalg.norm(right_wrench.linear) > 10.0)
            total_wrench = useLeft * left_wrench + useRight * right_wrench
            if (normal_force := total_wrench.linear[2]) > 0.0:
                CoP[0, i] = -total_wrench.angular[1] / normal_force
                CoP[1, i] = total_wrench.angular[0] / normal_force
                CoP[2, i] = (
                    (useLeft * rdata.oMf[lfId].translation[2] * left_wrench.linear[2]) + (useRight * rdata.oMf[rfId].translation[2] * right_wrench.linear[2])
                ) / normal_force

        LF[:, i] = se3_to_xyzrpy(left_foot)
        RF[:, i] = se3_to_xyzrpy(right_foot)
        LFvel[:, i] = left_foot_vel.vector
        RFvel[:, i] = right_foot_vel.vector
        for frame in frames:
            frameTransforms[frame][:, i] = se3_to_xyzrpy(rdata.oMf[rmodel.getFrameId(frame)])
        if leftFootPolygon is not None:
            for j in range(len(leftFootPolygon)):
                LFPolygon[j, :, i] = se3_to_xyzrpy(rdata.oMf[rmodel.getFrameId(leftFootPolygon[j])])
        if rightFootPolygon is not None:
            for j in range(len(rightFootPolygon)):
                RFPolygon[j, :, i] = se3_to_xyzrpy(rdata.oMf[rmodel.getFrameId(rightFootPolygon[j])])
    DCM = CoM + dCoM / np.sqrt(9.81 / CoM[2, :])

    # Get dbsp regions
    n_contacts = np.array(n_contacts)
    _n_contacts = n_contacts
    _n_contacts[_n_contacts > 1] = 2
    dbsp_start = np.concatenate([np.array([0]), np.where((_n_contacts[1:] - _n_contacts[:-1]) > 0)[0] + 1])
    dbsp_end = np.concatenate((np.where((_n_contacts[1:] - _n_contacts[:-1]) < 0)[0], [len(t) - 2])) + 1
    dbsp = [(dbsp_start[i], dbsp_end[i]) for i in range(len(dbsp_start))]

    def get_figure(
        n_rows,
        n_cols,
        data,
        ids,
        label=None,
        lb=None,
        ub=None,
        mid=None,
        stairs=False,
        types=None,
        colors=None,
    ):
        fig, axes = plt.subplots(n_rows, n_cols, sharex=True)
        if len(axes.shape) == 1:
            axes = [axes]
        fig.suptitle(label)

        def plot(y, style="solid", color="k", labels=None, legend=""):
            if stairs:
                [
                    axes[i // n_cols][i % n_cols].stairs(
                        y[k][: len(t)],
                        t[: len(y[k]) + 1],
                        label=str(labels[i]) + legend if labels is not None else None,
                        linestyle=style,
                        color=color,
                    )
                    for i, k in enumerate(ids)
                ]
            else:
                [
                    axes[i // n_cols][i % n_cols].plot(
                        t[: len(y[k])],
                        y[k][: len(t)],
                        label=str(labels[i]) + legend if labels is not None else None,
                        linestyle=style,
                        color=color,
                    )
                    for i, k in enumerate(ids)
                ]

        if types is not None:
            for i in range(len(data)):
                plot(
                    data[i],
                    labels=ids,
                    legend=types[i],
                    color=colors[i] if colors is not None else "k",
                )
        else:
            plot(data, labels=ids)
        if bounds:
            if lb is not None:
                plot(lb, style="--", color="r")
            if ub is not None:
                plot(ub, style="--", color="r")
            if mid is not None:
                plot(mid, style=":")
        [axes[i // n_cols][i % n_cols].legend() for i in range(n_rows * n_cols)]
        [axes[i // n_cols][i % n_cols].grid() for i in range(n_rows * n_cols)]
        [axes[i // n_cols][i % n_cols].axvspan(t[d[0]], t[d[1]], alpha=0.5, color="k") for d in dbsp for i in range(n_rows * n_cols)]
        plt.subplots_adjust(top=0.936, bottom=0.054, left=0.033, right=0.99, hspace=0.2, wspace=0.162)
        return fig

    # Plotting the joint positions, velocities and torques
    figs = []

    if comTargets is not None:
        figs.append(
            get_figure(
                1,
                2,
                [CoP[:2, :], DCM[:2, :], CoM[:2, :], comTargets[:2, :]],
                range(2),
                "CoM/DCM/CoP positions [m]",
                types=[" CoP", " DCM", " CoM", " target"],
                colors=["m", "b", "k", "g"],
            )
        )
    else:
        figs.append(
            get_figure(
                1,
                2,
                [CoP[:2, :], DCM[:2, :], CoM[:2, :]],
                range(2),
                "CoM/DCM/CoP positions [m]",
                types=[" CoP", " DCM", " CoM"],
                colors=["m", "b", "k"],
            )
        )
    window.add_figure("CoM/DCM/CoP", figs[-1])

    figs.append(get_figure(3, 4, X, range(7, 19), "Joint position [rad]", X_LB, X_UB, X_MID))
    window.add_figure("joint position", figs[-1])

    figs.append(get_figure(3, 4, X, range(nq + 6, nq + 18), "Joint velocity [rad/s]", X_LB, X_UB))
    window.add_figure("joint velocity", figs[-1])

    figs.append(get_figure(3, 4, U, range(0, nu), "Joint torque [Nm]", U_LB, U_UB, stairs=True))
    window.add_figure("joint torque", figs[-1])

    for label, foot, footData in zip(["LF", "RF"], [leftFoot, rightFoot], [LF, RF]):
        id = rmodel.getFrameId(foot)
        figs.append(
            get_figure(
                2,
                3,
                ([footData, frameTargets[id]] if id in frameTargets.keys() else [footData]) if frameTargets is not None else [footData],
                range(6),
                f"{label} position [m]",
                types=[" sol", " target"],
                colors=["k", "g"],
            )
        )
        window.add_figure(f"{label} position", figs[-1])

    figs.append(get_figure(2, 3, [LFvel], range(6), "LF position [m.s^-1]", types=[" sol"], colors=["k"]))
    window.add_figure("LF velocity", figs[-1])

    figs.append(get_figure(2, 3, [RFvel], range(6), "RF velocity [m.s^-1]", types=[" sol"], colors=["k"]))
    window.add_figure("RF velocity", figs[-1])

    for name, data in frameTransforms.items():
        id = rmodel.getFrameId(name)
        figs.append(
            get_figure(
                2,
                3,
                ([data, frameTargets[id]] if id in frameTargets.keys() else [data]) if frameTargets is not None else [data],
                list(range(6)),
                f"{name} position [m]",
                types=[" sol", " target"],
                colors=["k", "g"],
            )
        )
        window.add_figure(f"{name} position", figs[-1])

    if clearanceFrames is not None:
        for label, side, foot, footData in zip(["LF", "RF"], ["left", "right"], [leftFoot, rightFoot], [LF, RF]):
            figs.append(plt.figure())
            plt.plot(footData[0, :], footData[2, :])
            if frameTargets is not None:
                if (id := rmodel.getFrameId(foot)) in frameTargets.keys():
                    plt.plot(frameTargets[id][0, :], frameTargets[id][2, :], color="g")
            [
                plt.plot(x, y, color="k")
                for x, y in zip(
                    list(
                        np.vstack(
                            (
                                frameTransforms[clearanceFrames[side][0]][0, :],
                                frameTransforms[clearanceFrames[side][1]][0, :],
                            )
                        ).T
                    ),
                    list(
                        np.vstack(
                            (
                                frameTransforms[clearanceFrames[side][0]][2, :],
                                frameTransforms[clearanceFrames[side][1]][2, :],
                            )
                        ).T
                    ),
                )
            ]
            plt.suptitle(f"{label} clearance")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.grid(True)
            window.add_figure(f"{label} clearance", figs[-1])

    for forces in [left_forces, right_forces]:
        for name, data in forces.items():
            if name not in contact_frames:
                continue
            figs.append(get_figure(2, 3, np.array(data).T, list(range(6)), f"{name} force [N]"))
            window.add_figure(f"{name} force", figs[-1])

    fig6 = plt.figure()
    plt.plot(CoP[0, :], CoP[1, :], label="CoP", color="m")
    plt.plot(DCM[0, :], DCM[1, :], label="DCM", color="b")
    plt.plot(CoM[0, :], CoM[1, :], label="CoM", color="k")
    if leftFootPolygon is not None:
        for i in range(1, len(left_contact)):
            if left_contact[i]:
                fig6.axes[0].add_patch(patches.Polygon(LFPolygon[:, :2, i], closed=True, fill=False, color="k"))
    if rightFootPolygon is not None:
        for i in range(1, len(right_contact)):
            if right_contact[i]:
                fig6.axes[0].add_patch(patches.Polygon(RFPolygon[:, :2, i], closed=True, fill=False, color="k"))
    plt.suptitle("CoM/DCM/CoP position XY")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()
    window.add_figure("com position XY", fig6)

    fig6 = plt.figure()
    plt.plot(LF[0, :], LF[1, :])
    plt.suptitle("Left Foot position XY")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    window.add_figure("LF position XY", fig6)

    fig7 = plt.figure()
    plt.plot(RF[0, :], RF[1, :])
    plt.suptitle("Right Foot position XY")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    window.add_figure("RF position XY", fig7)

    if show:
        window.show()
        TabbedFigure.exec_qt_app(qt_app)
