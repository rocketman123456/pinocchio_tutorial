from casadi import *
import casadi

x = MX.sym("x", 2)  # States
p = MX.sym("p")  # Free parameter
u = MX.sym("u", 1)  # Controls

# Van der Pol oscillator system  (nonlinear ODE)
ode = vertcat((1.0 - x[1] * x[1]) * x[0] - x[1] + u, x[0])

T = 10  # Time horizon
N = 20  # Number of control intervals

# Integrator to discretize the system
intg_options = {"tf": T / N, "simplify": True, "number_of_finite_elements": 4}

# DAE problem structure
dae = {"x": x, "p": u, "ode": ode}

intg = integrator("intg", "rk", dae, 0, T / N)
# res = intg([0, 1], 0, [], [], [], [], [])
# res = intg(x0=[0, 1], p=0)
# print(res["xf"])
# f = Function("f", [x, y], [x, sin(y) * x], ["x", "y"], ["r", "q"])

# system integration
x = [0, 1]  # Initial state
for k in range(4):
    # Integrate 1s forward in time:
    # call integrator symbolically
    res = intg(x0=x, p=0)
    x = res["xf"]
    print(x)

# multiple shooting example
opti = casadi.Opti()

x = opti.variable(2, N + 1)  # Decision variables for state trajetcory
u = opti.variable(1, N)
p = opti.parameter(2, 1)  # Parameter (not optimized over)

opti.minimize(sumsqr(x) + sumsqr(u))

for k in range(N):
    next = intg(x0=x[:, k], p=u[:, k])
    opti.subject_to(x[:, k + 1] == next["xf"])

opti.subject_to(opti.bounded(-1.0, u, 1.0))
opti.subject_to(x[:, 1] == p)

opti.solver("sqpmethod")  # {"qpsol", "qrqp"}

# set init value
opti.set_value(p, [0, 1])

sol = opti.solve()
print(sol)
