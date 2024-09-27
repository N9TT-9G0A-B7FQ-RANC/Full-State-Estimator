import casadi as ca

x = ca.SX.sym('x')
y = ca.SX.sym('y')

# Objective function: f(x, y) = (x - 1)^2 + (y - 2)^2
objective = (x - 1)**2 + (y - 2)**2

# Constraint: g(x, y) = x^2 + y^2 - 1 <= 0
g = x**2 + y**2 - 1

nlp = {
    'x': ca.vertcat(x, y),   # Optimization variables (x, y)
    'f': objective,          # Objective function
    'g': g                   # Constraints
}

# Using the 'ipopt' solver
solver = ca.nlpsol('solver', 'ipopt', nlp)

###########################
# Initial guess for (x, y)
x0 = [0.5, 0.5]

# Bounds on the variables
lbx = [-ca.inf, -ca.inf]  # No lower bounds
ubx = [ca.inf, ca.inf]    # No upper bounds

# Bounds on the constraints (inequality constraint g(x, y) <= 0)
lbg = -ca.inf   # No lower bound on g
ubg = 0         # Upper bound g(x, y) <= 0

# Solve the NLP
solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

import numpy as np

# Extract the optimized values of (x, y)
x_opt = np.asarray(solution['x'])


print(f'Optimal solution: x = {x_opt[0]}, y = {x_opt[1]}')
print(f'Optimal objective value: f = {solution["f"]}')
