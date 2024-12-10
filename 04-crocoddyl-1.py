import numpy as np
import crocoddyl
import matplotlib.pylab as plt
from unicycle_utils import plotUnicycleSolution

x = np.random.rand(3)
u = np.random.rand(2)

# Unicycle dynamical model
v, w = u
c, s = np.cos(x[2]), np.sin(x[2])
dt = 1e-2
dx = np.array([v * c, v * s, w])
xnext = x + dx * dt

# Cost function: driving to origin (state) and reducing speed (control)
stateWeight = 1
ctrlWeight = 1
costResiduals = np.concatenate([stateWeight * x, ctrlWeight * u])
cost = 0.5 * sum(costResiduals**2)

model = crocoddyl.ActionModelUnicycle()
data = model.createData()

model.costWeights = np.array(
    [
        1,  # state weight
        1,  # control weight
    ]
)

### HYPER PARAMS: horizon and initial state
T = 100
x0 = np.array([-1, -1, 1])

problem = crocoddyl.ShootingProblem(x0, [model] * T, model)

us = [np.array([1.0, 0.1]).T for t in range(T)]
xs = problem.rollout(us)

plotUnicycleSolution(xs)
plt.axis([-3, 1.0, -2.0, 2.0])

# Select the solver for this problem
ddp = crocoddyl.SolverDDP(problem)

# Add solvers for verbosity and plots
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

done = ddp.solve()
assert done

plotUnicycleSolution(ddp.xs)
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.show()

log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
crocoddyl.plotConvergence(
    log.costs,
    log.pregs,
    log.dregs,
    log.grads,
    log.stops,
    log.steps,
    figIndex=2,
    show=False,
)

print(ddp.xs[-1])
