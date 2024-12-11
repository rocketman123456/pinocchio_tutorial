# %load tp3/unicycle_toward_origin.py
import crocoddyl
import numpy as np
import matplotlib.pylab as plt
import unittest
from unicycle_utils import plotUnicycleSolution

### HYPER PARAMS: horizon and initial state
T = 100
x0 = np.array([0.3, -10, 0])

### PROBLEM DEFINITION

model = crocoddyl.ActionModelUnicycle()
model_term = crocoddyl.ActionModelUnicycle()

model.costWeights = np.array([1e-4, 1]).T  # state weight  # control weight
model_term.costWeights = np.array([10, 0]).T  # state weight  # control weight

# Define the optimal control problem.
problem = crocoddyl.ShootingProblem(x0, [model] * T, model_term)
# Select the solver for this problem
ddp = crocoddyl.SolverDDP(problem)
ddp.th_stop = 1e-15
# Add solvers for verbosity and plots
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

### SOLVE THE PROBLEM
done = ddp.solve([], [], maxiter=1000)
# assert done
print(f"done : {done}")

### PLOT
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

plotUnicycleSolution(log.xs)
plt.show()

print("Type plt.show() to display the result.")


### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class UnicycleTest(unittest.TestCase):
    def test_logs(self):
        print(self.__class__.__name__)
        self.assertTrue(len(ddp.xs) == len(ddp.us) + 1)
        self.assertTrue(np.allclose(ddp.xs[0], ddp.problem.x0))
        self.assertTrue(ddp.stop < 1e-6)


if __name__ == "__main__":
    UnicycleTest().test_logs()
