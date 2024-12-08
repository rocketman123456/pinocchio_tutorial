from casadi import *
from matplotlib.pyplot import plot, show

x = MX.sym("x", 2)  # Two states
p = MX.sym("p")  # Free parameter

# Expression for ODE right-hand side
z = 1 - x[1] ** 2
rhs = vertcat(z * x[0] - x[1] + 2 * tanh(p), x[0])
print(rhs)

# ODE declaration with free parameter
ode = {"x": x, "p": p, "ode": rhs}

# Construct a Function that integrates over 1s
F = integrator("F", "cvodes", ode, 0, 1)

# Control vector
u = MX.sym("u", 4, 1)

x = [0, 1]  # Initial state
for k in range(4):
    # Integrate 1s forward in time:
    # call integrator symbolically
    res = F(x0=x, p=u[k])
    x = res["xf"]


# NLP declaration
nlp = {"x": u, "f": dot(u, u), "g": x}

# Solve using IPOPT
solver = nlpsol("solver", "ipopt", nlp)
res = solver(x0=0.2, lbg=0, ubg=0)

plot(res["x"])
show()
