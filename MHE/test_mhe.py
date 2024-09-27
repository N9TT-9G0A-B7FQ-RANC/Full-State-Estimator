import unittest
import numpy as np
from utils import MHE
import casadi as ca

class TestStageCostFunction(unittest.TestCase):

    def setUp(self):

        m = 1.0     # Mass (kg)
        k = 10.0    # Spring stiffness (N/m)
        c = 0.5     # Damping coefficient (Ns/m)

        # Define time
        T = 10.0    # Total time for simulation (seconds)
        dt = 0.01   # Time step (seconds)
        N = int(T / dt)  # Number of time steps

        # Define the state variables
        x = ca.SX.sym('x')    # Position
        v = ca.SX.sym('v')    # Velocity

        # State vector
        states = ca.vertcat(x, v)

        # Define the state derivatives
        x_dot = v
        v_dot = -(c/m)*v - (k/m)*x  # Equation of motion: m*ddot(x) + c*dot(x) + k*x = 0

        # State derivative vector
        rhs = ca.vertcat(x_dot, v_dot)

        ode = {'x': states, 'ode': rhs}

        self.initial_P = np.diag((0.1, 0.1))
        self.initial_Q = np.diag((0.1, 0.1))
        self.initial_R = np.diag((0.1, 0.1))

        self.past_horizon = 10

        self.obj = MHE(ode, dt, self.past_horizon, self.initial_P, self.initial_Q, self.initial_R)

    def test_arrival_cost(self):

        x = np.ones(2)
        x_bar = np.zeros(2)
        P = np.diag((1, 1))

        cost = self.obj.arrival_cost(x, x_bar, P)
        expected_cost = 2

        self.assertAlmostEqual(cost, expected_cost, places=5)

    def test_stage_cost(self):

        # Test with zero disturbances
        w = np.ones((2, self.past_horizon))
        x = np.zeros((2, self.past_horizon))
        x_bar = np.zeros((2, self.past_horizon))
        
        P = np.diag((1, 1))
        Q = np.diag((1, 1))

        cost = self.obj.stage_cost(x, x_bar, w, P, Q)

        expected_cost = 0.

        self.assertAlmostEqual(cost, expected_cost, places=5)

if __name__ == '__main__':
    unittest.main()