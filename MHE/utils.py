import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

class MHE:

    def __init__(
            self,
            f, 
            g,
            dt:float,
            horizon:int,
            Q:np.array,
            R:np.array):

        self.horizon = horizon

        # Dynamical system 
        self.f = f

        # Measurement function 
        self.g = g

        # State matrix
        self.Q = Q
        self.Q_inv = ca.inv(Q)

        # Measurement matrix
        self.R = R 
        self.R_inv = ca.inv(R)

        self.dt = dt  

        self.df__dx = self.get_symbolic_gradient_df__dx()
        self.df__dw = self.get_symbolic_gradient_df__dw()
        self.dg__dx = self.get_symbolic_gradient_dg__dx()
        
        self.one_step_integrator = self.get_symbolic_integrator()
        self.integrator = self.get_symbolic_integration_of_f()

        self.arrival_cost = self.get_symbolic_arrival_cost()
        self.stage_cost = self.get_symbolic_stage_cost()

        x = ca.DM(np.ones((2, self.horizon)))
        x_meas = ca.DM(np.zeros((2, self.horizon)))
        w = ca.DM(np.zeros((2, self.horizon)))
        pi_inv = ca.DM(np.ones((2, 2)))
        q_inv = ca.DM(np.ones((2, 2)))

        self.loss_function = self.get_symbolic_loss_function()

    def update_covariance_matrix(
            self, x, u, d,
            P: ca.SX, 
            Q: ca.SX, 
            R: ca.SX) -> ca.SX:
        """
        Update the covariance matrix.

        Parameters:
        x : ca.SX
            State vector.
        u : ca.SX
            Control input vector.
        d : ca.SX
            Disturbance vector.
        P : ca.SX
            Covariance matrix.
        Q : ca.SX
            Process noise covariance matrix.
        R : ca.SX
            Measurement noise covariance matrix.

        Returns:
        ca.SX
            Updated covariance matrix.
        """

        assert P.shape[0] == P.shape[1]

        A0 = self.df__dx(x, u, d)
        G0 = self.df__dw(x, u, d)
        C0 = self.dg__dx(x, u, d)

        TMP1 = Q @ G0 @ G0.T
        TMP2 = P @ A0 @ A0.T

        TMP3 = ca.inv(R + Q @ C0 @ C0.T)
        TMP4 = P @ A0 @ C0.T

        TMP5 = TMP4 @ TMP3 @ TMP4.T

        return TMP1 + TMP2 - TMP5
    
    def get_symbolic_arrival_cost(self):

        def arrival_cost(
            x:ca.SX, 
            x_bar:ca.SX, 
            P_inv:ca.SX
            ) -> ca.SX:
            """
            x = estimated x (CasADi SX)
            x_bar = previous estimated x (CasADi SX)
            P_inv = covariance matrix inverse (CasADi SX)
            """
            delta_x = x - x_bar
            return ca.mtimes(delta_x.T, ca.mtimes(P_inv, delta_x))
        
        x0 = ca.MX.sym('x0', 2, 1)
        x0_bar = ca.MX.sym('x0_bar', 2, 1)
        P = ca.MX.sym('P', 2, 2)
        return ca.Function('arrival_cost', [x0, x0_bar, P], [arrival_cost(x0, x0_bar, P)])

    def get_symbolic_stage_cost(self):

        def stage_cost(x, x_meas, w, Q_inv, R_inv):
            """
            x : current state estimate
            x_meas : a priori state estimate
            w : disturbances
            pi_inv : associated covariance to the a priori state estimate
            q_inv : covariance disturbance
            """
            delta_x = x - x_meas
            cum_sum = 0
            for i in range(delta_x.shape[0]):
                cum_sum += ca.mtimes(delta_x[:, i].T, ca.mtimes(R_inv, delta_x[:, i]))
            return cum_sum
        
        x = ca.MX.sym('x0', 2, self.horizon)
        x_meas = ca.MX.sym('x0_bar', 2, self.horizon)
        w = ca.MX.sym('d', 2, self.horizon) 
        R_inv = ca.MX.sym('P', 2, 2)
        Q_inv = ca.MX.sym('Q', 2, 2)
        return ca.Function('stage_cost', [x, x_meas, w, Q_inv, R_inv], [stage_cost(x, x_meas, w, Q_inv, R_inv)])
    
    def get_symbolic_gradient_df__dx(self):
        u_in = ca.MX.sym('u_in', 2, 1)
        d_in = ca.MX.sym('d_in', 2, 1)
        x_in = ca.MX.sym('x_in', 2, 1)
        jacobian_x = ca.diag(ca.jacobian(self.f(x_in, u_in, d_in), x_in))
        return ca.Function('grad_df_dx', [x_in, u_in, d_in], [jacobian_x])
    
    def get_symbolic_gradient_df__dw(self):
        u_in = ca.MX.sym('u_in', 2, 1)
        d_in = ca.MX.sym('d_in', 2, 1)
        x_in = ca.MX.sym('x_in', 2, 1)
        jacobian_d = ca.diag(ca.jacobian(self.f(x_in, u_in, d_in), d_in))
        return ca.Function('grad_df_dw', [x_in, u_in, d_in], [jacobian_d])
    
    def get_symbolic_gradient_dg__dx(self):
        u_in = ca.MX.sym('u_in', 2, 1)
        d_in = ca.MX.sym('d_in', 2, 1)
        x_in = ca.MX.sym('x_in', 2, 1)
        jacobian_x = ca.diag(ca.jacobian(self.g(x_in, u_in, d_in), x_in))
        return ca.Function('grad_dg_dx', [x_in, u_in, d_in], [jacobian_x])
    
    def get_symbolic_integrator(self):
        u_in = ca.DM(2, 1) #MX.sym('u_in', 2, 1)
        d_in = ca.DM(2, 1) #MX.sym('d_in', 2, 1)
        x_in = ca.MX.sym('x_in', 2, 1)
        # dae = {'x': x_in, 'p': ca.vertcat(u_in, d_in), 'ode': self.f(x_in, u_in, d_in)}
        # opts = {'tf': self.dt}
        # one_step_integrator = ca.integrator('integrator', 'cvodes', dae, opts)
        # res = one_step_integrator(x0=x_in, p=ca.vertcat(u_in, d_in))
        # x_next = res['xf']
        f = self.f(x_in, u_in, d_in)  # f should return the derivative dx/dt

        # Euler integration step
        x_next = x_in + f * self.dt  # Update state based on the Euler method
        return ca.Function('f_next_state', [x_in, u_in, d_in], [x_next])

    def get_symbolic_integration_of_f(self):

        def integrate_f_over_past_horizon(x0, u, d):
            x = x0
            x_out = ca.MX(2, self.horizon)
            for i in range(self.horizon):
                x_out[:, i] = self.g(x, u[:, i], d[:, i])
                x = self.one_step_integrator(x, u[:, i], d[:, i])
            return x_out
        
        x_in = ca.MX.sym('x_in', 2, 1)
        u_in = ca.MX.sym('u_in', 2, self.horizon)
        d_in = ca.MX.sym('d_in', 2, self.horizon)

        return ca.Function('f_integration', [x_in, u_in, d_in], [integrate_f_over_past_horizon(x_in, u_in, d_in)])
    
    def get_symbolic_loss_function(self):

        def loss_function(x_est, x_bar, x_meas, u, d, P_inv, Q_inv, R_inv):
            x_pred = self.integrator(x_est, u, d)
            return self.arrival_cost(x_est, x_bar, P_inv) + self.stage_cost(x_pred, x_meas, d, Q_inv, R_inv)
        
        x_est = ca.MX.sym('x_est', 2, 1)
        x_bar = ca.MX.sym('x_bar', 2, 1)
        x_meas = ca.MX.sym('x_meas', 2, self.horizon)
        u_in = ca.MX.sym('u_in', 2, self.horizon)
        d_in = ca.MX.sym('d_in', 2, self.horizon)
        P_inv = ca.MX.sym('P_inv', 2, 2)
        Q_inv = ca.MX.sym('Q_inv', 2, 2)
        R_inv = ca.MX.sym('Q_inv', 2, 2)

        return ca.Function('loss_function', [x_est, x_bar, x_meas, u_in, d_in, P_inv, Q_inv, R_inv], [loss_function(x_est, x_bar, x_meas, u_in, d_in, P_inv, Q_inv, R_inv)])
    
    def optimize(self, x_bar, x_meas, u_in, d_in, P):

        P_inv = ca.inv(P)

        x_est = ca.MX.sym('x_est', 2, 1)
    
        # Define the NLP problem
        nlp = {
            'x': x_est,
            'f': self.loss_function(x_est, x_bar, x_meas, u_in, d_in, P_inv, self.Q_inv, self.R_inv),
            'g': []  # Add constraints here if needed
        }

        # Create the solver instance

        opts={}
        opts["ipopt"] = {"max_iter": 1, "tol": 1e-10, "print_level": 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        lbx = [-ca.inf, -ca.inf]  # Lower bounds on x
        ubx = [ca.inf, ca.inf]  # Upper bounds on x

        # Solve the problem
        sol = solver(x0=x_bar, lbx=lbx, ubx=ubx)
        # if solver.stats()['success']:
        x_opt = sol['x'].full()
        # else:
            # print("Optimization failed!")
        # # Extract the solution
        # x_opt = sol['x'].full()
        # print(f"Optimal solution: x_est = {x_opt}")

        # Predict next state
        x_opt_next = self.one_step_integrator(x_opt, u[:, 0], d[:, 0])

        # Update covariance matrix
        P_est = self.update_covariance_matrix(x_opt, u[:, 0], d[:, 0], P, self.Q, self.R)
        print(np.diag(P_est))
        return x_opt, x_opt_next, P_est, sol['f']

if __name__ == '__main__':

    # Define time
    T = 10.0    # Total time for simulation (seconds)
    dt = 0.02   # Time step (seconds)
    N = int(T / dt)  # Number of time steps

    # Define the system dynamics function
    def f(x, u, d):
        x0 = -0.1 * x[0] + 0 * u[0] + 0 * d[0]
        x1 = -0.2 * x[1] + x[0] + 0 * u[1] + 0 * d[1]
        return ca.vertcat(x0, x1)
    
    def g(x, u ,d):
        x0 = 1 * x[0]
        x1 = 0 * x[1]
        return ca.vertcat(x0, x1)
    
    # Generate measurements
    x0 = np.ones((2, 1))
    u = np.zeros((2, N))
    d = np.zeros((2, N))
    x_meas = []
    x = x0
    for i in range(N):
        x_meas.append(x + np.random.randn(2, 1) * 0.1)
        x = f(x, u[:, i], d[:, i]) * dt + x 
    x_meas = np.asarray(x_meas).T[0]

    sigma_p = 1
    P0 = np.diag((sigma_p**2, sigma_p**2))

    sigma_q = 1
    Q = np.diag((sigma_q**2, sigma_q**2))

    sigma_r = 0.05
    R = np.diag((sigma_r**2, sigma_r**2))

    # MHE loop
    past_horizon = 5

    mhe = MHE(f, g, dt, past_horizon, Q, R)

    x_meas = ca.DM(x_meas)

    x_bar = np.asarray([1, 1])
    x_opt_list = []
    sol_f_list = []
    for i in range(N-past_horizon):
        xopt, x_bar, P0, sol_f = mhe.optimize(x_bar, x_meas[:, i:i+past_horizon], u[:, i:i+past_horizon], d[:, i:i+past_horizon], P0)
        x_opt_list.append(xopt)
        sol_f_list.append(sol_f)
    print()

    # plt.plot(np.asarray(res)[:, 0])
    plt.plot(np.asarray(x_meas)[0])

    print()