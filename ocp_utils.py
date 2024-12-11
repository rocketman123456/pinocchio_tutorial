import casadi
import pinocchio.casadi as cpin
import numpy as np

SX_zeros = lambda n: casadi.SX(np.zeros(n))


def frameAcceleration(cmodel, cdata, cq, cv, ca, index, update_kinematics=True, ref_frame=cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
    if update_kinematics:
        cpin.forwardKinematics(cmodel, cdata, cq, cv, ca)
    a_local = cpin.getFrameClassicalAcceleration(cmodel, cdata, index)
    if ref_frame == cpin.ReferenceFrame.LOCAL:
        return a_local

    H = cdata.oMf[index]
    if ref_frame == cpin.ReferenceFrame.WORLD:
        a_world = H.act(a_local)
        return a_world

    Hr = cpin.SE3(H.rotation, SX_zeros(3))
    a = Hr.act(a_local)
    return a


def updateFrames(cmodel, cdata, cq, cv):
    cpin.forwardKinematics(cmodel, cdata, cq, cv)
    cpin.framesForwardKinematics(cmodel, cdata, cq)


def framePlacement(cmodel, cdata, cq, cv, frameIndex, referenceFrame=cpin.ReferenceFrame.WORLD):
    updateFrames(cmodel, cdata, cq, cv)
    return cdata.oMf[frameIndex]


def framePlacementFunctions(cmodel, cdata, cq, cv, frameIndex, reference_frame=cpin.ReferenceFrame.WORLD):
    """
    Given a frame will output the functions for obtaining the position and rotation from the state
    """
    symbolicRep = framePlacement(cmodel, cdata, cq, cv, frameIndex, reference_frame)
    x = casadi.vertcat(cq, cv)
    rotFunction = casadi.Function(f"R_{frameIndex}", [x], [symbolicRep.rotation], ["x"], [f"R_{frameIndex}"])
    posFunction = casadi.Function(f"p_{frameIndex}", [x], [symbolicRep.translation], ["x"], [f"p_{frameIndex}"])
    return rotFunction, posFunction


def frameVelocity(cmodel, cdata, cq, cv, foot_frame_Id, reference_frame=cpin.ReferenceFrame.WORLD):
    updateFrames(cmodel, cdata, cq, cv)
    return cpin.getFrameVelocity(cmodel, cdata, foot_frame_Id, reference_frame)


def symSE3(rotation, translation):
    motion = cpin.SE3()
    motion.translation = casadi.SX(translation)
    motion.rotation = casadi.SX(rotation)
    return motion


def symMotion(angular, linear):
    motion = cpin.Motion()
    motion.linear = casadi.SX(linear)
    motion.angular = casadi.SX(angular)
    return motion


def framePlacementFunctions(cmodel, cdata, cq, cv, frameIndex, reference_frame=cpin.ReferenceFrame.WORLD):
    """
    Given a frame will output the functions for obtaining the position and rotation from the state
    """
    symbolicRep = framePlacement(cmodel, cdata, cq, cv, frameIndex, reference_frame)
    x = casadi.vertcat(cq, cv)
    rotFunction = casadi.Function(f"R_{frameIndex}", [x], [symbolicRep.rotation], ["x"], [f"R_{frameIndex}"])
    posFunction = casadi.Function(f"p_{frameIndex}", [x], [symbolicRep.translation], ["x"], [f"p_{frameIndex}"])
    return rotFunction, posFunction


def frameVelocityFunctions(cmodel, cdata, cq, cv, frameIndex, reference_frame=cpin.ReferenceFrame.WORLD):
    """
    Given a frame will output the functions for obtaining the position and rotation from the state
    """
    symbolicRep = frameVelocity(cmodel, cdata, cq, cv, frameIndex, reference_frame)
    x = casadi.vertcat(cq, cv)
    linear = casadi.Function(f"R_{frameIndex}", [x], [symbolicRep.linear], ["x"], [f"R_{frameIndex}"])
    angular = casadi.Function(f"p_{frameIndex}", [x], [symbolicRep.angular], ["x"], [f"p_{frameIndex}"])
    return linear, angular


def framePlacementFunctionsRedundant(cmodel, cdata, cq, cv, nu, frameIndex, reference_frame=cpin.ReferenceFrame.WORLD):
    """
    Given a frame will output the functions for obtaining the position and rotation from the state
    """
    cq_ = casadi.vertcat(cq, cq[-int(nu / 2) :])
    cv_ = casadi.vertcat(cv, cv[-int(nu / 2) :])
    symbolicRep = framePlacement(cmodel, cdata, cq_, cv_, frameIndex, reference_frame)
    x = casadi.vertcat(cq, cv)
    rotFunction = casadi.Function(f"R_{frameIndex}", [x], [symbolicRep.rotation], ["x"], [f"R_{frameIndex}"])
    posFunction = casadi.Function(f"p_{frameIndex}", [x], [symbolicRep.translation], ["x"], [f"p_{frameIndex}"])
    return rotFunction, posFunction


def frameVelocityFunctionsRedundant(cmodel, cdata, cq, cv, nu, frameIndex, reference_frame=cpin.ReferenceFrame.WORLD):
    """
    Given a frame will output the functions for obtaining the position and rotation from the state
    """
    cq_ = casadi.vertcat(cq, cq[-int(nu / 2) :])
    cv_ = casadi.vertcat(cv, cv[-int(nu / 2) :])
    symbolicRep = frameVelocity(cmodel, cdata, cq_, cv_, frameIndex, reference_frame)
    x = casadi.vertcat(cq, cv)
    linear = casadi.Function(f"R_{frameIndex}", [x], [symbolicRep.linear], ["x"], [f"R_{frameIndex}"])
    angular = casadi.Function(f"p_{frameIndex}", [x], [symbolicRep.angular], ["x"], [f"p_{frameIndex}"])
    return linear, angular


def framePlacementCost(framePlacement, refPlacement, weight=np.ones(6)):

    e = casadi.SX(np.zeros(6))

    R_err = refPlacement.rotation.T @ framePlacement.rotation

    e[:3] = framePlacement.translation - refPlacement.translation
    e[3:] = cpin.log3(R_err)

    return e.T @ np.diag(weight) @ e


def framePositionCost(framePlacement, refPlacement, weight=np.ones(3)):

    e = framePlacement.translation - refPlacement.translation

    return e.T @ np.diag(weight) @ e


def frameRotationCost(framePlacement, refPlacement, weight=np.ones(3)):

    R_err = refPlacement.rotation.T @ framePlacement.rotation

    e = cpin.log3(R_err)

    return e.T @ np.diag(weight) @ e


def frameVelocityCost(frameVelocity, refVelocity, weight=np.ones(6)):

    e = casadi.SX(np.zeros(6))

    e[:3] = frameVelocity.linear - refVelocity.linear
    e[3:] = frameVelocity.angular - refVelocity.angular

    return e.T @ np.diag(weight) @ e


def mechanicalEnergyCostFunction(cmodel, cdata, cq, cv):
    mechEnergy = cpin.computeMechanicalEnergy(cmodel, cdata, cq, cv)
    x = casadi.vertcat(cq, cv)
    mechEnergyFunction = casadi.Function("Mech_Energy", [x], [mechEnergy], ["x"], ["state"])
    return mechEnergyFunction
