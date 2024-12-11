import numpy as np
import pinocchio

import crocoddyl


def opposite_side(side):
    if side == "left":
        return "right"
    elif side == "right":
        return "left"
    else:
        raise ValueError(f"[Side] Unknown side {side}")


class SimpleBipedGaitProblem:
    """Build simple bipedal locomotion problems.

    This class aims to build simple locomotion problems used in the examples of
    Crocoddyl.
    The scope of this class is purely for academic reasons, and it does not aim to be
    used in any robotics application.
    We also do not consider it as part of the API, so changes in this class will not
    pass through a strict process of deprecation.
    Thus, we advice any user to DO NOT develop their application based on this class.
    """

    def __init__(
        self,
        rmodel,
        q0,
        flyngSide,
        rightFoot,
        leftFoot,
        options,
        rightToe=None,
        leftToe=None,
        integrator="euler",
        control="zero",
        fwddyn=True,
        no_impulse=False,
    ):
        """Construct biped-gait problem.

        :param rmodel: robot model
        :param rightFoot: name of the right foot
        :param leftFoot: name of the left foot
        :param integrator: type of the integrator
            (options are: 'euler', and 'rk4')
        :param control: type of control parametrization
            (options are: 'zero', 'one', and 'rk4')
        """
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.options = options
        self.constraint_opts = self.options["constraint"]
        self.tracking_opts = self.options["tracking"]
        self.reg_opts = self.options["regularization"]
        # Getting the frame id for all the legs
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        self.rToeId = self.rmodel.getFrameId(rightToe) if rightToe is not None else self.rfId
        self.lToeId = self.rmodel.getFrameId(leftToe) if leftToe is not None else self.lfId
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn
        self.no_impulse = no_impulse
        self.flying_side = flyngSide
        # Defining default state
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)

    def get_initial_state(self, x0):
        q0 = x0[: self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = pinocchio.SE3(self.rdata.oMf[self.rfId])
        lfPos0 = pinocchio.SE3(self.rdata.oMf[self.lfId])
        return rfPos0, lfPos0

    def createUpstairsProblem(
        self,
        x0,
        stepLength,
        stepHeight,
        extraStepHeight,
        stairsHeight,
        toeClearance,
        timeStep,
        stepKnots,
        supportKnots,
    ):
        """Create a shooting problem for a simple walking gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param stairsHeight: stairs height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        rfPos0, lfPos0 = self.get_initial_state(x0)
        feetPos = {self.rfId: rfPos0, self.lfId: lfPos0}

        # Defining the action models along the time instances
        loco3dModel = []
        feetTargets = {i: [] for i in [self.rfId, self.lfId]}

        def append_step_with_double_support(
            side,
            step_length,
            extra_step_height=0.0,
            stair_climbing=False,
            toe_x_limit=None,
            cop_x_offset=0.0,
            foot_y_offset=0.0,
        ):
            nonlocal loco3dModel, feetTargets, feetPos
            # Create double support phase
            doubleSupport = []
            for k in range(supportKnots):
                dbsp_model, dbsp_feetTarget = self.createSwingFootModelUpstairs(timeStep, {id: pos for id, pos in feetPos.items()})
                doubleSupport.append(dbsp_model)
                for i in feetTargets.keys():
                    if i in dbsp_feetTarget.keys():
                        feetTargets[i].append(dbsp_feetTarget[i])
                    else:
                        feetTargets[i].append(np.ones(6) * np.nan)

            if side == "left":
                swingFootTask = {self.lfId: feetPos[self.lfId]}
                supportFootTask = {self.rfId: feetPos[self.rfId]}
                toeId = self.lToeId
            elif side == "right":
                swingFootTask = {self.rfId: feetPos[self.rfId]}
                supportFootTask = {self.lfId: feetPos[self.lfId]}
                toeId = self.rToeId
            else:
                raise ValueError(f"[biped] Unknown {side=}")

            # Get swing foot target and constraints
            swingToeConstraints = []
            if stair_climbing:
                # Create the swing foot target (function)
                get_swing_foot_translation = self.get_stairs_translation_function(
                    stepKnots,
                    step_length,
                    stepHeight,
                    extra_step_height,
                    stairsHeight,
                    foot_y_offset,
                )

                # Define the step collision-avoiding constraints
                if toe_x_limit is not None:
                    swingToeConstraints.append(
                        [
                            [-1.0, 0.45],  # Spv range
                            {
                                toeId: {
                                    "name": "step_collision",
                                    "lb": -np.Inf * np.ones(3),
                                    "ub": np.array([toe_x_limit, np.Inf, np.Inf]),
                                }
                            },
                        ]
                    )
                    swingToeConstraints.append(
                        [
                            [1.0 / 3.0, 2.0 / 3.0],  # Spv range
                            {
                                toeId: {
                                    "name": "clearance",
                                    "lb": np.array([-np.Inf, -np.Inf, stairsHeight + toeClearance]),
                                    "ub": np.array([np.Inf, np.Inf, np.Inf]),
                                }
                            },
                        ]
                    )
            else:
                # Create the swing foot target (function)
                get_swing_foot_translation = self.get_walk_translation_function(stepKnots, step_length, stepHeight, foot_y_offset)

            # Create single support phase nodes
            step_model, step_feet = self.createStepModels(
                get_swing_foot_translation,
                timeStep,
                stepKnots,
                supportFootTask,
                swingFootTask,
                swingToeConstraints=swingToeConstraints,
                cop_x_offset=cop_x_offset,
            )
            # # Updating the current foot position for next step
            # for id in feetPos.keys():
            #     feetPos[id] = pinocchio.XYZQUATToSE3(step_feet[id][-1])

            # Append double support and step models
            loco3dModel += doubleSupport + step_model

            # Append logs
            for id in feetTargets.keys():
                if id in step_feet.keys():
                    feetTargets[id] += step_feet[id]
                else:
                    feetTargets[id] += [[np.ones(6) * np.nan] * (stepKnots + 1)]

        # Creating the action models for stair climbing
        # Left step
        append_step_with_double_support(
            self.flying_side,
            stepLength,
            extraStepHeight,
            stair_climbing=True,
            toe_x_limit=0.24,
            foot_y_offset=-0.05 if self.flying_side == "left" else 0.05,
        )

        # Right step
        append_step_with_double_support(
            opposite_side(self.flying_side),
            stepLength + 0.15,
            extraStepHeight,
            stair_climbing=True,
            toe_x_limit=0.24,
            cop_x_offset=0.03,
        )

        # Add a stopping step
        append_step_with_double_support(self.flying_side, 0.15, foot_y_offset=0.05 if self.flying_side == "left" else -0.05)

        # Final double support domain
        joint_reg = self.reg_opts["joint"]
        joint_final = self.options["final"]["joint"]
        for k in range(supportKnots):

            # Set special values for terminal node (standstill continuity)
            terminalModel = k == supportKnots - 1
            jointNeutralWeight = joint_final["position"] if terminalModel else joint_reg["position"]
            jointVelWeight = joint_final["velocity"] if terminalModel else joint_reg["velocity"]
            velBound = 0.0 if terminalModel else np.Inf

            model, feetTarget = self.createSwingFootModelUpstairs(
                timeStep,
                {id: pos for id, pos in feetPos.items()},
                jointNeutralWeight=jointNeutralWeight,
                jointVelWeight=jointVelWeight,
                velBound=velBound,
            )

            # Append node
            loco3dModel.append(model)

            # Append logs
            for id in feetTargets.keys():
                if id in feetTarget.keys():
                    feetTargets[id].append(feetTarget[id])
                else:
                    feetTargets[id].append(np.ones(6) * np.nan)

        # Stack targets (logs)
        for id in feetTargets.keys():
            feetTargets[id] = np.vstack(feetTargets[id]).T

        # Rescale the terminal weights (Necessary ?)
        if hasattr(loco3dModel[-1], "differential"):
            costs = loco3dModel[-1].differential.costs.costs.todict()
        else:
            costs = loco3dModel[-1].costs.costs.todict()
        for c in costs.values():
            c.weight *= timeStep

        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem, feetTargets, None

    def get_stairs_translation_function(self, stepKnots, step_length, stepHeight, extra_step_height, stairsHeight, foot_y_offset=0.0):
        def get_swing_foot_translation(k):
            # Defining a foot swing task given the step length, stairs height, step height and extra step height.
            dp = np.zeros(3)
            dp[1] = foot_y_offset / (stepKnots + 1)
            if k < int(stepKnots / 3):
                dp[2] = (stepHeight + extra_step_height + stairsHeight) / int(stepKnots / 3)
            elif k <= 2 * int(stepKnots / 3):
                dp[0] = step_length / (int(stepKnots / 3) + 1)
            else:
                dp[2] = -(stepHeight + extra_step_height) / (int(stepKnots / 3) + 2)
            return dp

        return get_swing_foot_translation

    def get_walk_translation_function(self, stepKnots, step_length, stepHeight, foot_y_offset=0.0):

        def get_swing_foot_translation(k):
            # Defining a foot swing task given the step length and step height.
            dp = np.zeros(3)

            dp[0] = step_length / (stepKnots + 1)
            dp[1] = foot_y_offset / (stepKnots - 1)

            if k < int(stepKnots / 2):
                dp[2] = stepHeight / int(stepKnots / 2)
            else:
                dp[2] = -stepHeight / int(stepKnots / 2)

            return dp

        return get_swing_foot_translation

    def createStepModels(
        self,
        get_swing_foot_translation,
        timeStep,
        numKnots,
        supportFoot,
        swingFoot,
        swingToeConstraints=[],
        cop_x_offset=0.0,
    ):
        """Action models for a footstep phase.

        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFoot: Task of the support foot
        :param supportFoot: Task of the swing foot
        :return footstep action models
        """
        feetTargets = {id: [] for id in list(swingFoot.keys()) + list(supportFoot.keys())}

        # Action models for the foot swing
        footSwingModel = []
        swing_tracking = self.tracking_opts["swingfoot"]
        for k in range(numKnots + 1):
            spv = float(k) / numKnots
            swingFootTask = {}
            for i, p in swingFoot.items():
                # Create swing foot target
                dp = get_swing_foot_translation(k)
                p.translation += dp

                # Change weight dynamically for smooth double support transition (max weight should be double support weight)
                weight = swing_tracking["min_weight"] + (swing_tracking["max_weight"] - swing_tracking["min_weight"]) * np.abs(k - numKnots / 2) / numKnots / 2
                swingFootTask[i] = {
                    "p": p,  # Constant attitude
                    "w": weight,
                }

            constraints = [cstr for (spv_min, spv_max), cstr in swingToeConstraints if (spv >= spv_min) and (spv <= spv_max)] if swingToeConstraints else []
            model, feetTarget = self.createSwingFootModelUpstairs(
                timeStep,
                supportFoot,
                swingFootTask=swingFootTask,
                constraints=constraints,
                cop_x_offset=cop_x_offset,
            )
            footSwingModel += [model]
            for i in feetTargets.keys():
                if i in feetTarget.keys():
                    feetTargets[i].append(feetTarget[i])
                else:
                    feetTargets[i].append(np.ones(6) * np.nan)

        # Action model for the foot switch (impact)
        footSwitchModel, feetTarget = self.createFootSwitchModel(timeStep, supportFoot, swingFootTask, pseudoImpulse=self.no_impulse)
        for i in feetTargets.keys():
            if i in feetTarget.keys():
                feetTargets[i].append(feetTarget[i])
            else:
                feetTargets[i].append(np.ones(6) * np.nan)

        return [*footSwingModel, footSwitchModel], feetTargets

    def createSwingFootModelUpstairs(
        self,
        timeStep,
        supportFootTask,
        swingFootTask={},
        jointNeutralWeight=0.01,
        jointVelWeight=100.0,
        constraints=[],
        cop_x_offset=0.0,
        velBound=np.Inf,
    ):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootTask: Task of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        feetTarget = {}
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootTask.keys())
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i, p in supportFootTask.items():
            feetTarget[i] = pinocchio.SE3ToXYZQUAT(p).copy()
            self._add_contact(i, feetTarget[i], contactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        for i, p in supportFootTask.items():
            feetTarget[i] = pinocchio.SE3ToXYZQUAT(p)

            # Place the feet
            self._add_foot_tracking_cost(i, feetTarget[i], self.tracking_opts["supportfoot"]["weight"], costModel)

            # Wrench cone (no slippage)
            self._add_wrench_cone_limit(i, costModel)

            # CoP box (safety margin)
            # self._add_cop_box_limit(i, p.rotation, cop_x_offset, costModel)

        for i, params in swingFootTask.items():
            feetTarget[i] = pinocchio.SE3ToXYZQUAT(params["p"])

            # Move the feet
            self._add_foot_tracking_cost(
                i,
                feetTarget[i],
                params["w"],
                costModel,
                self.tracking_opts["swingfoot"]["ratio"],
            )

            # Move (X,Y,R,P,Y) only if Z high enough
            # self._add_fly_high_cost(i, costModel)

            # Penalize vertical velocity (usually disabled)
            self._add_vert_velocity_cost(i, self.reg_opts["swingfoot"]["verticalvelocity"], costModel)

        if len(constraints) > 0:
            [self._add_constraint_limit(id, params, costModel) for constraint in constraints for id, params in constraint.items()]

        stateWeights = np.array(
            [0.0] * 3 + [self.reg_opts["freeflyer"]["orientation"]] * 3 + [jointNeutralWeight] * (self.state.nv - 6) + [jointVelWeight] * self.state.nv
        )

        model = self.get_model_with_joint_bounds(timeStep, contactModel, costModel, nu, stateWeights, velBound=velBound)

        return model, feetTarget

    def _add_contact(self, i, target, contactModel):
        supportContactModel = crocoddyl.ContactModel6D(
            self.state,
            i,
            pinocchio.XYZQUATToSE3(target),
            pinocchio.LOCAL,
            contactModel.nu,
            np.array(self.options["contact"]["baumgarte"]["gains"]),
        )
        contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

    def _add_com_cost(self, comTask, costs):
        comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, costs.nu)
        comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
        costs.addCost("comTrack", comTrack, 1e3)

    def _add_foot_tracking_cost(self, i, footTarget, weight, costs, weightRatio=np.ones(6)):
        if np.isclose(weight, 0.0):
            return
        framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i, pinocchio.XYZQUATToSE3(footTarget), costs.nu)
        activation = crocoddyl.ActivationModelWeightedQuad(np.array(weightRatio))
        footTrack = crocoddyl.CostModelResidual(self.state, activation, framePlacementResidual)
        costs.addCost(self.rmodel.frames[i].name + "_footTrack", footTrack, weight)

    def _add_foot_damping_cost(self, i, weight, costs):
        if np.isclose(weight, 0.0):
            return
        frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
            self.state,
            i,
            pinocchio.Motion.Zero(),
            pinocchio.LOCAL,
            costs.nu,
        )
        impulseFootVelCost = crocoddyl.CostModelResidual(self.state, frameVelocityResidual)
        costs.addCost(
            self.rmodel.frames[i].name + "_impulseVel",
            impulseFootVelCost,
            weight,
        )

    def _add_fly_high_cost(self, i, costs):
        if np.isclose(self.tracking_opts["swingfoot"]["flyhigh"]["weight"], 0.0):
            return
        flyHighResidual = crocoddyl.ResidualModelFlyHigh(self.state, i, self.tracking_opts["swingfoot"]["flyhigh"]["slope"], costs.nu)
        activation = crocoddyl.ActivationModelQuad(2)
        footFlyHigh = crocoddyl.CostModelResidual(self.state, activation, flyHighResidual)
        costs.addCost(
            self.rmodel.frames[i].name + "_footFlyHigh",
            footFlyHigh,
            self.tracking_opts["swingfoot"]["flyhigh"]["weight"],
        )

    def _add_vert_velocity_cost(self, i, weight, costs):
        if np.isclose(weight, 0.0):
            return
        vertical_velocity_reg_residual = crocoddyl.ResidualModelFrameVelocity(
            self.state,
            i,
            pinocchio.Motion.Zero(),
            pinocchio.ReferenceFrame.WORLD,
            costs.nu,
        )
        vertical_velocity_activation = crocoddyl.ActivationModelWeightedQuad(np.array([0, 0, 1, 0, 0, 0]))

        vertical_velocity_reg_cost = crocoddyl.CostModelResidual(self.state, vertical_velocity_activation, vertical_velocity_reg_residual)
        costs.addCost(self.rmodel.frames[i].name + "_vel_zReg", vertical_velocity_reg_cost, weight)

    def _add_constraint_limit(self, id, params, costs):
        if np.isclose(self.constraint_opts["frametranslation"]["weight"], 0.0):
            return
        toePlacementResidual = crocoddyl.ResidualModelFrameTranslation(self.state, id, np.zeros(3), costs.nu)
        bounds = crocoddyl.ActivationBounds(params["lb"], params["ub"], 1.0)
        activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
        footTrack = crocoddyl.CostModelResidual(self.state, activation, toePlacementResidual)
        costs.addCost(
            self.rmodel.frames[id].name + "_" + params["name"],
            footTrack,
            self.constraint_opts["frametranslation"]["weight"],
        )

    def _add_wrench_cone_limit(self, i, costs):
        if np.isclose(self.constraint_opts["wrenchcone"]["weight"], 0.0):
            return
        cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.1, 0.05]))
        wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(self.state, i, cone, costs.nu, self._fwddyn)
        wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
        wrenchCone = crocoddyl.CostModelResidual(self.state, wrenchActivation, wrenchResidual)
        costs.addCost(
            self.rmodel.frames[i].name + "_wrenchCone",
            wrenchCone,
            self.constraint_opts["wrenchcone"]["weight"],
        )

    # def _add_cop_box_limit(self, i, R, cop_x_offset, costs):
    #     copSupport = crocoddyl.CoPSupport(
    #         R, np.array([cop_x_offset, 0.0]), np.array(self.constraint_opts["copbox"]["size"])
    #     )
    #     residualCoPPosition = crocoddyl.ResidualModelContactCoPPosition(
    #         self.state, i, copSupport, costs.nu
    #     )
    #     copActivation = crocoddyl.ActivationModelQuadraticBarrier(
    #         crocoddyl.ActivationBounds(copSupport.lb, copSupport.ub)
    #     )
    #     copBox = crocoddyl.CostModelResidual(self.state, copActivation, residualCoPPosition)
    #     costs.addCost(
    #         self.rmodel.frames[i].name + "_copBox", copBox, self.constraint_opts["copbox"]["weight"]
    #     )

    def get_model_with_joint_bounds(
        self,
        timeStep,
        contactModel,
        costModel,
        nu,
        stateWeights,
        impulseModel=None,
        velBound=np.Inf,
    ):
        # Regularization costs
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        costModel.addCost("stateReg", stateReg, self.reg_opts["state"]["position"])
        if impulseModel is None:
            accelerationResidual = crocoddyl.ResidualModelJointAcceleration(self.state, np.zeros(18), nu)
            accelReg = crocoddyl.CostModelResidual(self.state, accelerationResidual)
            costModel.addCost("accelReg", accelReg, self.reg_opts["joint"]["acceleration"])
            if self._fwddyn:
                ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
            else:
                ctrlResidual = crocoddyl.ResidualModelJointEffort(self.state, self.actuation, nu)
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
            if not np.isclose(self.reg_opts["joint"]["torque"], 0):
                costModel.addCost("ctrlReg", ctrlReg, self.reg_opts["joint"]["torque"])

        # Joint position and velocity limits
        xlb = np.concatenate(
            [
                -np.Inf * np.ones(6),  # dimension of the SE(3) manifold
                self.state.lb[7:19],
                -velBound * np.ones(self.state.nv),
            ]
        )
        xub = np.concatenate(
            [
                np.Inf * np.ones(6),  # dimension of the SE(3) manifold
                self.state.ub[7:19],
                velBound * np.ones(self.state.nv),
            ]
        )
        # Frontal Hip
        xlb[6] += np.deg2rad(5.0)  # Required
        xub[12] -= np.deg2rad(5.0)  # Required
        xlb[12] += np.deg2rad(5.0)  # Not required but helps convergence rate
        xub[6] -= np.deg2rad(5.0)  # Not required but helps convergence rate
        # Sagittal Knee
        xlb[9] += np.deg2rad(5.0)  # Required
        xlb[15] += np.deg2rad(5.0)  # Required
        xub[9] -= np.deg2rad(5.0)  # Not required but helps convergence rate
        xub[15] -= np.deg2rad(5.0)  # Not required but helps convergence rate

        bounds = crocoddyl.ActivationBounds(xlb, xub, self.constraint_opts["joint"]["limit"]["r"])
        xLimitResidual = crocoddyl.ResidualModelState(self.state, nu)
        constraintManager = crocoddyl.ConstraintModelManager(self.state, nu)
        if not self.options["constraint"]["hard"]["use"]:
            xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
            limitCost = crocoddyl.CostModelResidual(self.state, xLimitActivation, xLimitResidual)
            if not np.isclose(self.constraint_opts["joint"]["limit"]["weight"], 0):
                costModel.addCost("xLimitCost", limitCost, self.constraint_opts["joint"]["limit"]["weight"])

        else:
            # Hard constraints have to impact
            #  xLimitResidual = crocoddyl.ResidualModelState(self.state, np.zeros(self.state.nx), nu)
            #  residualConstraint = crocoddyl.ConstraintModelResidual(self.state, xLimitResidual,self.state.lb, self.state.ub)
            residualConstraint = crocoddyl.ConstraintModelResidual(self.state, xLimitResidual, xlb, xub)
            constraintManager.addConstraint("xLimitConstraint", residualConstraint)

        if impulseModel is None:
            # Ctrl limits
            xlb = -self.rmodel.effortLimit[6:]
            xub = self.rmodel.effortLimit[6:]

            bounds = crocoddyl.ActivationBounds(xlb, xub, self.constraint_opts["torque"]["limit"]["r"])
            xLimitResidual = crocoddyl.ResidualModelControl(self.state, nu)
            xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
            limitCost = crocoddyl.CostModelResidual(self.state, xLimitActivation, xLimitResidual)
            if not np.isclose(self.constraint_opts["torque"]["limit"]["weight"], 0):
                costModel.addCost("ctrlLimitCost", limitCost, self.constraint_opts["torque"]["limit"]["weight"])

        if impulseModel is not None:
            model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel, constraintManager, 0.0, 0.0, True)
        else:
            # Creating the action model for the KKT dynamics with an integration scheme
            if self._fwddyn:
                dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                    self.state,
                    self.actuation,
                    contactModel,
                    costModel,
                    constraintManager,
                    0.0,
                    True,
                )
            else:
                dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                    self.state,
                    self.actuation,
                    contactModel,
                    costModel,
                    constraintManager,
                )
            # Use integration scheme to convert differential (continuous) to discrete dynamics (the simpler the faster)
            if self._control == "one":
                control = crocoddyl.ControlParametrizationModelPolyOne(nu)
            elif self._control == "rk4":
                control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.four)
            elif self._control == "rk3":
                control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.three)
            else:
                control = crocoddyl.ControlParametrizationModelPolyZero(nu)
            if self._integrator == "euler":
                model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
            elif self._integrator == "rk2":
                model = crocoddyl.IntegratedActionModelRK(dmodel, control, crocoddyl.RKType.two, timeStep)
            elif self._integrator == "rk3":
                model = crocoddyl.IntegratedActionModelRK(dmodel, control, crocoddyl.RKType.three, timeStep)
            elif self._integrator == "rk4":
                model = crocoddyl.IntegratedActionModelRK(dmodel, control, crocoddyl.RKType.four, timeStep)

        return model

    def createFootSwitchModel(self, timeStep, supportFootTask, swingFootTask, pseudoImpulse=True):
        """Action model for a foot switch phase.

        :param supportFootTask: Task of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the
            impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(timeStep, supportFootTask, swingFootTask)
        else:
            return self.createImpulseModel(supportFootTask, swingFootTask)

    def createPseudoImpulseModel(self, timeStep, supportFootTask, swingFootTask):
        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
        """

        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        feetTarget = {}
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootTask)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i, p in supportFootTask.items():
            feetTarget[i] = pinocchio.SE3ToXYZQUAT(p).copy()
            self._add_contact(i, feetTarget[i], contactModel)

        # Get costs
        costModel, stateWeights = self.get_foot_switch_costs(supportFootTask, swingFootTask, nu, feetTarget)

        # Create node
        model = self.get_model_with_joint_bounds(timeStep, contactModel, costModel, nu, stateWeights)

        return model, feetTarget

    def createImpulseModel(self, supportFootTask, swingFootTask):
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param supportFootTask: Task of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        feetTarget = {}
        nu = 0
        # Creating a 6D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootTask.keys():
            supportContactModel = crocoddyl.ImpulseModel6D(self.state, i, pinocchio.LOCAL)
            impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)
        for i in swingFootTask.keys():
            supportContactModel = crocoddyl.ImpulseModel6D(self.state, i, pinocchio.LOCAL)
            impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)

        # Get costs
        costModel, stateWeights = self.get_foot_switch_costs(supportFootTask, swingFootTask, nu, feetTarget)

        # Create node
        model = self.get_model_with_joint_bounds(0.0, impulseModel, costModel, nu, stateWeights, impulseModel)

        return model, feetTarget

    def get_foot_switch_costs(self, supportFootTask, swingFootTask, nu, feetTarget):
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        for i, p in supportFootTask.items():
            feetTarget[i] = pinocchio.SE3ToXYZQUAT(p)

            # Place the feet
            self._add_foot_tracking_cost(i, feetTarget[i], self.tracking_opts["supportfoot"]["weight"], costModel)

            # Wrench cone (no slippage)
            self._add_wrench_cone_limit(i, costModel)

        for i, params in swingFootTask.items():
            feetTarget[i] = pinocchio.SE3ToXYZQUAT(params["p"])

            # Place the feet
            self._add_foot_tracking_cost(i, feetTarget[i], self.tracking_opts["swingfoot"]["max_weight"], costModel)

            # Stop swing foot (super high velocity cost)
            self._add_foot_damping_cost(i, self.options["impact"]["swingfoot"]["weight"], costModel)

        joint_reg = self.reg_opts["joint"]
        stateWeights = np.array(
            [0.0] * 3
            + [self.reg_opts["freeflyer"]["orientation"]] * 3
            + [joint_reg["position"]] * (self.state.nv - 6)
            + [joint_reg["velocity"]] * self.state.nv
        )

        return costModel, stateWeights
