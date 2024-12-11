import numpy as np
import pinocchio as pin
import time
from meshcat_viewer_wrapper import MeshcatVisualizer


class Visual:
    """
    Class representing one 3D mesh to be attached to a joint
    """

    def __init__(self, name, jointParent, placement):
        self.name = name  # Name in gepetto viewer
        self.jointParent = jointParent  # ID (int) of the joint
        self.placement = placement  # placement of the body wrt joint, i.e. bodyMjoint

    def place(self, display, oMjoint, refresh=True):
        oMbody = oMjoint * self.placement
        display.applyConfiguration(self.name, pin.SE3ToXYZQUATtuple(oMbody))
        if refresh:
            display.refresh()


class LeggedRobot(object):
    """
    Class following robot-wrapper template, with a model, a data and a viewer object.
    Display with robot.display. Initial configuration set to self.q0.
    """

    def __init__(
        self,
        nbJoint,
        nLegs,
        baseWidth=1.0,
        baseLength=2.0,
        baseHeight=0.5,
        linkLength=1.0,
        floatingMass=1.0,
        linkMass=1.0,
        loadModel=True,
        useViewer=True,
        baseType="euler3d",
        RX=True,
        leg_placements=None,
    ):
        self.sceneName = "world"
        self.first_display = True
        self.nbJoint = nbJoint
        self.nLegs = nLegs
        self.baseWidth = baseWidth
        self.baseLength = baseLength
        self.baseHeight = baseHeight
        self.floatingMass = floatingMass
        self.linkLength = linkLength
        self.linkMass = linkMass
        self.loadModel = loadModel
        self.useViewer = useViewer
        self.baseType = baseType
        self.RX = RX
        self.leg_placements = leg_placements
        self.buildLeggedRobot()

    def buildLeggedRobot(self):
        """Build a quadruped system"""
        robot = pin.robot_wrapper.RobotWrapper()
        robot.initViewer(sceneName=self.sceneName)

        if self.useViewer:
            self.viewer = viewer = robot.viewer
            # self.viewer = MeshcatVisualizer(robot, url=None)
        self.model = rmodel = pin.Model()
        self.visuals = visuals = []

        self.colors = {"white": [1, 1, 1, 1.0], "black": [0, 0, 0, 1.0], "grey": [0.8, 0.8, 0.8, 1.0], "red": [183 / 256, 28 / 256, 28 / 256, 1.0]}

        jointId = 0
        jointPlacement = pin.SE3.Identity()

        baseId = self.addBase(jointId, jointPlacement)

        if self.leg_placements is None:
            self.initLegPlacements()

        leg_placement = pin.SE3.Identity()
        self.growLegs(leg_placement, baseId)

        self.setLimits()
        self.setInitialPosition()

        if self.useViewer and self.loadModel:
            self.viewer.gui.addFloor("world/floor")
            self.viewer.gui.setLightingMode("world/floor", "ON")
            self.viewer.gui.setCameraTransform(
                "python-pinocchio", [0, -5 * self.nbJoint * self.linkLength, 1 * self.nbJoint * self.linkLength / 2] + [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]
            )

        self.data = rmodel.createData()

    def addBase(self, jointId, jointPlacement):
        if self.baseType == "prismatic":
            baseInertia = pin.Inertia(self.floatingMass, np.array([0.0, 0.0, 0.0]), np.diagflat([1e-6, 1e-6, 1e-6]))
        elif self.baseType == "free_flyer" or self.baseType == "euler" or self.baseType == "euler3d":
            baseInertia = pin.Inertia(self.floatingMass, np.array([0.0, 0.0, 0.0]), np.diagflat([1e-2, 1e-2, 1e-2]))
        else:
            raise Warning("Base joint type not supported")

        if self.baseType == "prismatic":
            jointName, bodyName = ["prismatic_joint", "mass"]
            baseId = self.model.addJoint(jointId, pin.JointModelPZ(), jointPlacement, jointName)
        elif self.baseType == "free_flyer":
            jointName, bodyName = ["free_flyer", "mass"]
            baseId = self.model.addJoint(jointId, pin.JointModelFreeFlyer(), jointPlacement, jointName)
        elif self.baseType == "euler":
            jointName, bodyName = ["free_flyer", "mass"]
            jointId = self.model.addJoint(jointId, pin.JointModelPX(), jointPlacement, jointName)
            jointId = self.model.addJoint(jointId, pin.JointModelPZ(), jointPlacement, jointName)
            baseId = self.model.addJoint(jointId, pin.JointModelRY(), jointPlacement, jointName)
        elif self.baseType == "euler3d":
            jointName, bodyName = ["free_flyer", "mass"]
            jointId = self.model.addJoint(jointId, pin.JointModelPX(), jointPlacement, jointName)
            jointId = self.model.addJoint(jointId, pin.JointModelPY(), jointPlacement, jointName)
            jointId = self.model.addJoint(jointId, pin.JointModelPZ(), jointPlacement, jointName)
            jointId = self.model.addJoint(jointId, pin.JointModelRX(), jointPlacement, jointName)
            jointId = self.model.addJoint(jointId, pin.JointModelRY(), jointPlacement, jointName)
            baseId = self.model.addJoint(jointId, pin.JointModelRZ(), jointPlacement, jointName)
        else:
            raise Warning("Cannot parse the specified base joint type")

        self.model.addFrame(pin.Frame("base", baseId, 0, jointPlacement, pin.FrameType.OP_FRAME))
        self.model.appendBodyToJoint(baseId, baseInertia, pin.SE3.Identity())

        if self.useViewer and self.loadModel:
            self.viewer.gui.addBox(self.sceneName + "/" + bodyName, self.baseLength, self.baseWidth, self.baseHeight, self.colors["white"])
        self.visuals.append(Visual(self.sceneName + "/" + bodyName, baseId, pin.SE3.Identity()))

        return baseId

    def initLegPlacements(self):
        symmetric = not bool(self.nLegs % 2)
        nLegOnSide = int(self.nLegs / 2)
        x = self.baseLength * np.linspace(-1 / 2, 1 / 2, nLegOnSide)
        y = self.baseWidth * np.linspace(-1 / 2, 1 / 2, 2)
        xPositions, yPositions = np.meshgrid(x, y)
        zPositions = np.zeros(self.nLegs)
        self.leg_placements = np.array([[x, y, z] for (x, y, z) in zip(xPositions.flatten(), yPositions.flatten(), zPositions.flatten())])
        if not symmetric:
            try:
                self.leg_placements = np.vstack((np.zeros(3), self.leg_placements))
            except:
                # monoped case
                self.leg_placements = np.array(np.zeros((1, 3)))

    def growLegs(self, leg_placement, baseId):

        linkInertia = pin.Inertia(self.linkMass, np.array([0.0, 0.0, self.linkLength / 2]), self.linkMass / 5 * np.diagflat([1e-2, self.linkLength**2, 1e-2]))

        for leg in range(self.nLegs):
            leg_placement.translation = self.leg_placements[leg, :]
            # Revolute joints
            self.buildLeg(baseId, leg, leg_placement, linkInertia)

    def buildLeg(self, baseId, leg, leg_placement, linkInertia):
        for i in range(1, self.nbJoint + 1):
            jointName, bodyName = [f"leg_{leg}_revolute_joint_{i}", f"leg_{leg}_link_{i}"]
            if i == 1:
                if self.RX == True:
                    jointId = self.model.addJoint(baseId, pin.JointModelRX(), leg_placement, jointName)
                    self.model.addFrame(pin.Frame(f"leg_{leg}_RX_{i}", jointId, i - 1, pin.SE3.Identity(), pin.FrameType.JOINT))
                    jointId = self.model.addJoint(jointId, pin.JointModelRY(), pin.SE3.Identity(), jointName)
                else:
                    jointId = self.model.addJoint(baseId, pin.JointModelRY(), leg_placement, jointName)
                self.model.appendBodyToJoint(baseId, linkInertia, pin.SE3.Identity())
                jointPlacement = pin.SE3(np.eye(3), np.array([0.0, 0.0, self.linkLength]))
            else:
                jointPlacement = pin.SE3(np.eye(3), np.array([0.0, 0.0, self.linkLength]))
                jointId = self.model.addJoint(jointId, pin.JointModelRY(), jointPlacement, jointName)
                self.model.appendBodyToJoint(jointId, linkInertia, pin.SE3.Identity())
            self.model.addFrame(pin.Frame(f"leg_{leg}_RY_{i}", jointId, i - 1, pin.SE3.Identity(), pin.FrameType.JOINT))

            if self.useViewer and self.loadModel:
                self.viewer.gui.addSphere(self.sceneName + "/" + bodyName, self.linkLength / 12, self.colors["grey"])
                self.viewer.gui.addBox(
                    self.sceneName + "/" + bodyName + "_box", self.linkLength / 15, self.linkLength / 15, self.linkLength, self.colors["white"]
                )
            self.visuals.append(Visual(self.sceneName + "/" + bodyName, jointId, pin.SE3.Identity()))
            self.visuals.append(Visual(self.sceneName + "/" + bodyName + "_box", jointId, pin.SE3(np.eye(3), np.array([0, 0, self.linkLength / 2]))))

        self.model.addFrame(pin.Frame(f"foot_{leg}", jointId, 0, jointPlacement, pin.FrameType.OP_FRAME))
        if self.useViewer and self.loadModel:
            self.viewer.gui.addSphere(f"{self.sceneName}/{bodyName}_foot", self.linkLength / 12, self.colors["white"])
        self.visuals.append(Visual(f"{self.sceneName}/{bodyName}_foot", jointId, jointPlacement))

    def setLimits(self):
        if self.baseType == "prismatic":
            self.model.upperPositionLimit = np.concatenate((np.array([100]), 2 * np.pi * np.ones(self.nbJoint)), axis=0)
            self.model.lowerPositionLimit = np.concatenate((np.array([0.0]), -2 * np.pi * np.ones(self.nbJoint)), axis=0)
            self.model.velocityLimit = np.concatenate((np.array([100]), 5 * np.ones(self.nbJoint)), axis=0)
        else:
            self.model.upperPositionLimit = 1e3 * np.ones(self.model.nq)
            self.model.lowerPositionLimit = -1e3 * np.ones(self.model.nq)
            self.model.velocityLimit = 1e3 * np.ones(self.model.nq)

    def setInitialPosition(self):
        if self.RX:
            legs_q0 = ([0, -np.pi] + [0 for _ in range(self.nbJoint - 1)]) * self.nLegs
        else:
            legs_q0 = ([-np.pi] + [0 for _ in range(self.nbJoint - 1)]) * self.nLegs
        if self.baseType == "prismatic":
            self.q0 = np.array([self.nbJoint * self.linkLength] + legs_q0)
        elif self.baseType == "free_flyer":
            self.q0 = np.array([0, 0, self.nbJoint * self.linkLength, 0, 0, 0, 1] + legs_q0)
        elif self.baseType == "euler":
            self.q0 = np.array([0, self.nbJoint * self.linkLength, 0] + legs_q0)
        elif self.baseType == "euler3d":
            self.q0 = np.array([0, 0, self.nbJoint * self.linkLength, 0, 0, 0] + legs_q0)

    def display(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        if self.useViewer:
            for visual in self.visuals:
                visual.place(self.viewer.gui, self.data.oMi[visual.jointParent], False)
            self.viewer.gui.refresh()
            if self.first_display:
                self.first_display = False

    def animateDOFs(self, scaling_factor=1):
        self.display(self.q0)
        dt = 1e-3
        q = self.q0
        for i in range(self.model.nv):
            v = np.zeros([self.model.nv, 1])
            v[i] = 10
            n = 50
            for p in range(n):
                if p < n / 2:
                    q = pin.integrate(self.model, q, v * dt)
                else:
                    q = pin.integrate(self.model, q, -v * dt)

                self.display(q)
                time.sleep(dt * scaling_factor)


class Quadruped(LeggedRobot):
    def __init__(
        self,
        nbJoint=2,
        baseWidth=1.0,
        baseLength=2.0,
        baseHeight=0.5,
        linkLength=1.0,
        floatingMass=1.0,
        linkMass=1.0,
        loadModel=True,
        useViewer=True,
        baseType="euler3d",
        RX=True,
    ):
        super(Quadruped, self).__init__(nbJoint, 4, baseWidth, baseLength, baseHeight, linkLength, floatingMass, linkMass, loadModel, useViewer, baseType, RX)


class Monoped(LeggedRobot):
    def __init__(
        self,
        nbJoint=2,
        baseWidth=1.0,
        baseLength=2.0,
        baseHeight=0.5,
        linkLength=1.0,
        floatingMass=1.0,
        linkMass=1.0,
        loadModel=True,
        useViewer=True,
        baseType="euler3d",
        RX=True,
    ):
        super(Monoped, self).__init__(nbJoint, 1, baseWidth, baseLength, baseHeight, linkLength, floatingMass, linkMass, loadModel, useViewer, baseType, RX)


class Biped(LeggedRobot):
    def __init__(
        self,
        nbJoint=2,
        baseWidth=1.0,
        baseLength=0.25,
        baseHeight=0.5,
        linkLength=1.0,
        floatingMass=1.0,
        linkMass=1.0,
        loadModel=True,
        useViewer=False,
        baseType="euler3d",
        RX=True,
    ):
        super(Biped, self).__init__(
            nbJoint,
            2,
            baseWidth,
            baseLength,
            baseHeight,
            linkLength,
            floatingMass,
            linkMass,
            loadModel,
            useViewer,
            baseType,
            RX,
        )
        self.leg_placements[:, 0] = 0
        super(Biped, self).buildLeggedRobot()
