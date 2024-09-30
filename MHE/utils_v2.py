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
            P:np.array, 
            Q:np.array,
            R:np.array):

        self.horizon = horizon

        # State
        self.x = ca.MX.sym('x', 2, horizon)
        self.x0 = ca.MX.sym('x0', 2, 1)
        self.x_pred = ca.MX.sym('x_pred', 2, horizon)

        # Measurements
        # self.y = ca.SX.sym('y', 2, horizon)

        

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

        self.w = ca.SX.sym('w', 2, horizon)

        self.dt = dt  

        self.u = ca.MX.sym('u', 2, horizon)
        self.d = ca.MX.sym('d', 2, horizon)

        # Set gradient functions
        self.df__dx = self.get_symbolic_gradient_df__dx()
        self.df__dw = self.get_symbolic_gradient_df__dw()
        self.dg__dx = self.get_symbolic_gradient_dg__dx()

        # Set cost
        self.arrival_cost = self.get_symbolic_arrival_cost()
        self.stage_cost = self.get_symbolic_stage_cost()
        
        self.loss_function = self.get_symbolic_loss_function()

        self.constraint_function = self.get_symbolic_constraints()


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
            return (delta_x.T@P_inv)@delta_x
        
        x0 = ca.MX.sym('x0', 2, 1)
        x0_bar = ca.MX.sym('x0', 2, 1)
        P = ca.MX.sym('P', 2, 2)
        return ca.Function('arrival_cost', [x0, x0_bar, P], [arrival_cost(x0, x0_bar, P)])

    def get_symbolic_stage_cost(self):

        def stage_cost(x, x_meas, w, P_inv, Q_inv):
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
                cum_sum += (delta_x[:, i].T@P_inv)@delta_x[:, i]
            
            return cum_sum
        x = ca.MX.sym('x0', 2, self.horizon)
        x_bar = ca.MX.sym('x0', 2, self.horizon)
        w = ca.MX.sym('d', 2, self.horizon) 
        P = ca.MX.sym('P', 2, 2)
        Q = ca.MX.sym('P', 2, 2)
        return ca.Function('stage_cost', [x, x_bar, w, P, Q], [stage_cost(x, x_bar, w, P, Q)])
    
    def get_symbolic_gradient_df__dx(self):
        u_in = ca.MX.sym('u_in', 2, 1)
        d_in = ca.MX.sym('d_in', 2, 1)
        x_in = ca.MX.sym('x_in', 2, 1)
        # jacobian_x = ca.diag(ca.jacobian(self.f(self.x[:, 0], self.u[:, 0], self.d[:, 0]), self.x[:, 0]))
        # return ca.Function('grad_df_dx', [self.x[:, 0], self.u[:, 0], self.d[:, 0]], [jacobian_x])
        jacobian_x = ca.diag(ca.jacobian(self.f(x_in, u_in, d_in), x_in))
        return ca.Function('grad_df_dx', [x_in, u_in, d_in], [jacobian_x])
    
    def get_symbolic_gradient_df__dw(self):
        u_in = ca.MX.sym('u_in', 2, 1)
        d_in = ca.MX.sym('d_in', 2, 1)
        x_in = ca.MX.sym('x_in', 2, 1)
        # jacobian_x = ca.diag(ca.jacobian(self.f(self.x[:, 0], self.u[:, 0], self.d[:, 0]), self.d[:, 0]))
        # return ca.Function('grad_df_dw', [self.x[:, 0], self.u[:, 0], self.d[:, 0]], [jacobian_x])
        jacobian_d = ca.diag(ca.jacobian(self.f(x_in, u_in, d_in), d_in))
        return ca.Function('grad_df_dw', [x_in, u_in, d_in], [jacobian_d])
    
    def get_symbolic_gradient_dg__dx(self):
        u_in = ca.MX.sym('u_in', 2, 1)
        d_in = ca.MX.sym('d_in', 2, 1)
        x_in = ca.MX.sym('x_in', 2, 1)
        # jacobian_x = ca.diag(ca.jacobian(self.g(self.x[:, 0], self.u[:, 0], self.d[:, 0]), self.x[:, 0]))
        # return ca.Function('grad_dg_dx', [self.x[:, 0], self.u[:, 0], self.d[:, 0]], [jacobian_x])
        jacobian_x = ca.diag(ca.jacobian(self.g(x_in, u_in, d_in), x_in))
        return ca.Function('grad_dg_dx', [x_in, u_in, d_in], [jacobian_x])
    
    def one_step_integrator(self, x_in, u_in, d_in):
        # Set up the dynamic equation (DAE or ODE) for the system
        # Instead of symbolic inputs, we're using numeric values for x_in, u_in, and d_in
        x_sym = ca.MX.sym('x', x_in.shape[0])
        u_sym = ca.MX.sym('u', u_in.shape[0])
        d_sym = ca.MX.sym('d', d_in.shape[0])

        dae = {'x': x_sym, 'p': ca.vertcat(u_sym, d_sym), 'ode': self.f(x_sym, u_sym, d_sym)}

        opts = {'tf': self.dt}  # Set the time step for integration
        integrator = ca.integrator('integrator', 'cvodes', dae, opts)  # Create the CasADi integrator

        # Run the integrator for the given numeric inputs
        res = integrator(x0=x_in, p=ca.vertcat(u_in, d_in))

        x_next = res['xf']  # Get the full numeric value of the result, not a symbolic expression
        return x_next

    def integrator(self, x0, u, d):
        x = x0
        x_out = ca.DM(2, self.horizon)
        for i in range(self.horizon):
            x_out[:, i] = x
            x = self.one_step_integrator(x, u[:, i], d[:, i])
        return x_out
            
    def get_symbolic_loss_function(self):

        def loss_function(x_est, x_pred, x_bar, x_meas, u, d, P_inv, Q_inv):
            return self.arrival_cost(x_est, x_bar, P_inv) + self.stage_cost(x_pred, x_meas, d, P_inv, Q_inv)
        
        x_est = ca.MX.sym('x_est', 2, 1)
        x_pred = ca.MX.sym('x_pred', 2, self.horizon)
        x_bar = ca.MX.sym('x_bar', 2, 1)
        x_meas = ca.MX.sym('x_meas', 2, self.horizon)
        u_in = ca.MX.sym('u_in', 2, self.horizon)
        d_in = ca.MX.sym('d_in', 2, self.horizon)
        P_inv = ca.MX.sym('P_inv', 2, 2)
        Q_inv = ca.MX.sym('Q_inv', 2, 2)
        return ca.Function('loss_function', [x_est, x_pred, x_bar, x_meas, u_in, d_in, P_inv, Q_inv], [loss_function(x_est, x_pred, x_bar, x_meas, u_in, d_in, P_inv, Q_inv)])
    
    def get_symbolic_constraints(self):

        def constraint_function(x_pred, u, d):
            g = []  # List to store the constraints
            for i in range(self.horizon - 1):
                # Constraint: Difference between predicted next state and integrated next state
                constraint = x_pred[:, i + 1] - self.one_step_integrator(x_pred[:, i], u[:, i], d[:, i])
                g.append(ca.vec(constraint))  # Append constraint as a vector (2x1 flattened)

            # Vertically concatenate all constraints into a single vector
            g_all = ca.vertcat(*g)
            return g_all
        
        # Define the symbolic variable x_pred for the predicted states
        x_pred = ca.MX.sym('x_pred', 2, self.horizon)
        u = ca.MX.sym('u_in', 2, self.horizon-1)
        d = ca.MX.sym('d_in', 2, self.horizon-1)

        # Return the CasADi function representing the constraint
        return ca.Function('constraint', [x_pred, u, d], [constraint_function(x_pred, u, d)])
    
    def optimize(self, x_bar, x_meas, u_in, d_in, P):

        P_inv = np.linalg.inv(P)

        # Define the optimization problem
        x_est = ca.MX.sym('x_est', 2, 1)
       
        # Define initial guess and bounds
        lbx = [-ca.inf, -ca.inf]  # Lower bounds on x
        ubx = [ca.inf, ca.inf]  # Upper bounds on x

        # Integrate
        x_integrated = self.integrator(x_est, u_in, d_in)

        # Solve the problem
        # Create the solver instance
        # Define the NLP problem
        nlp = {
            'x': x_est,
            'f': self.loss_function(x_est, x_integrated, x_bar, x_meas, u_in, d_in, P_inv, self.Q_inv),
            'g': [] #self.constraint_function(x_integrated, u_in, d_in)
        }

        opts={}
        opts["ipopt"] = {"max_iter": 1, "tol": 1e-7,"print_level": 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x_bar, lbx=lbx, ubx=ubx)
        if solver.stats()['success']:
            x_opt = sol['x'].full()
        else:
            print("Optimization failed!")

        # Predict next state
        x_opt_next = self.one_step_integrator(x_opt, u[:, 0], d[:, 0])

        # Update covariance matrix
        P_est = self.update_covariance_matrix(x_opt, u[:, 0], d[:, 0], P, self.Q, self.R)

        return x_opt, x_opt_next, P_est

if __name__ == '__main__':

    # Define time
    T = 10.0    # Total time for simulation (seconds)
    dt = 0.01   # Time step (seconds)
    N = int(T / dt)  # Number of time steps

    # Define the system dynamics function
    def f(x, u, d):
        x0 = -0.1 * x[0] + 0 * u[0] + 0 * d[0]
        x1 = -0.2 * x[1] + 0 * u[1] + 0 * d[1]
        return ca.vertcat(x0, x1)
    
    def g(x, u ,d):
        x0 = 1 * x[0]
        x1 = 1 * x[1]
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

    # Initial conditions
    # x0 = 1.0  # Initial position (displaced by 1 unit)
    # v0 = 0.0  # Initial velocity
    # state_initial = np.array([x0, v0])

    # # Simulation results storage
    # trajectory = np.zeros((N, 2))  # To store [x, v] over time
    # time = np.linspace(0, T, N)    # Time vector

    # # Run the simulation
    # state = state_initial
    # for i in range(N):
    #     trajectory[i, :] = state + np.random.randn(2)*0.1
    #     result = integrator(x0=state)
    #     state = result['xf'].full().flatten()  # Extract the final state of this step

    # The higher it is, the smoother the output is
    P0 = np.diag((1000000, 1000000))
    Q = np.diag((10, 10))
    R = np.diag((10, 10))

    # MHE loop
    past_horizon = 100

    mhe = MHE(f, g, dt, past_horizon, P0, Q, R)

    # x_meas = mhe.integrator(x0, u, d) #+ np.random.randn(2, past_horizon) * 0.1
    x_meas = ca.DM(x_meas)

    x_bar = x_meas[:, 0]
    x_opt_list = []
    for i in range(N-past_horizon):
        xopt, x_bar, P0_ = mhe.optimize(x_bar, x_meas[:, i:i+past_horizon], u[:, i:i+past_horizon], d[:, i:i+past_horizon], P0)
        x_opt_list.append(xopt)
    print()

    # plt.plot(np.asarray(res)[:, 0])
    plt.plot(np.asarray(x_meas)[0])

    print()
        


        






















    # # Plot the results
    # plt.figure(figsize=(10, 5))
    # plt.subplot(2, 1, 1)
    # plt.plot(time, trajectory[:, 0], label='Position (x)')
    # plt.title('Spring-Mass-Damper System Simulation')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Position [m]')
    # plt.legend()
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(time, trajectory[:, 1], label='Velocity (v)', color='orange')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Velocity [m/s]')
    # plt.legend()
    # plt.grid()

    # plt.tight_layout()
    # plt.show()
