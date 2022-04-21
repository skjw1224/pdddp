import numpy as np
import casadi as ca
from functools import partial
from numpy.random import Generator, PCG64

class NQRconstEnv(object):
    def __init__(self, prob_type=None):
        self.name = "NQRconst"
        self.prob_type = prob_type

        # Dimensions and variables
        self.s_dim = 3
        self.a_dim = 1
        self.o_dim = 3

        self.param_real = np.array([[]]).T
        self.param_mu_prior = np.array([[]]).T
        self.param_sigma_prior = np.array([[]]).T
        self.p_dim = len(self.param_real)

        # Four types of constraints
        self.cPineq_dim, self.cPeq_dim = 2, 0 # Path constraint
        self.cTineq_dim, self.cTeq_dim = 1, 0  # Global terminal constraint

        self.c_dim = [self.cPineq_dim, self.cPeq_dim, self.cTineq_dim, self.cTeq_dim]

        # MX variable for dae function object (no SX)
        self.state_var = ca.MX.sym('x', self.s_dim)
        self.action_var = ca.MX.sym('u', self.a_dim)
        self.param_mu_var = ca.MX.sym('p_mu', self.p_dim)
        self.param_sigma_var = ca.MX.sym('p_sig', self.p_dim)
        self.param_epsilon_var = ca.MX.sym('p_eps', self.p_dim)
        self.ref_var = None

        self.real_env = False

        self.t0 = 0.
        self.dt = 0.5  # s
        self.tT = 9
        self.nT = int(self.tT / self.dt) + 1

        # self.x0 = np.array([[self.t0, -5, -5]]).T
        self.x0 = np.array([[self.t0, 1, 0]]).T
        self.u0 = np.array([[0.]]).T

        self.eps_t = np.finfo(np.float32).eps

        self.xmin = np.array([[self.t0, -2, -2]]).T
        self.xmax = np.array([[self.tT, 2, 2]]).T
        self.ymin = np.array([[self.t0, -2, -2]]).T
        self.ymax = np.array([[self.tT, 2, 2]]).T
        self.umin = np.array([[-2]]).T
        self.umax = np.array([[2]]).T
        self.gmin = np.array([[-1, -1]]).T
        self.gmax = np.array([[1, 1]]).T

        # Define noise related values
        self.init_state_idx = []
        self.state_noise_idx = [1, 2]
        self.meas_noise_idx = []
        self.init_state_noise = 0.02
        self.state_noise = 0.2
        self.meas_noise = 0.04

        self.zero_center_scale = True

        self.sym_expressions()
        self.model_derivs = self.eval_model_derivs()

        self.reset()

    def reset(self, x0=None):
        self.nprandom = Generator(PCG64(2022))

        if x0 is None:
            x0 = self.x0

        state = self.scale(x0, self.xmin, self.xmax)
        if self.prob_type == 'plant':
            # States with random initial values
            state[self.init_state_idx] = \
                state[self.init_state_idx] + self.nprandom.uniform(-self.init_state_noise, self.init_state_noise,
                                                                   [len(self.init_state_idx), 1])
            state = np.clip(state, -1, 1)

        time = self.t0

        p_mu, p_sigma, p_eps = self.param_real, np.zeros([self.p_dim, 1]), np.zeros([self.p_dim, 1])
        y = self.y_fnc(state, self.u0, p_mu, p_sigma, p_eps).full()
        return time, state, y

    def step(self, time, state, action, leg_BC=None, *args):
        # Scaled state, action, output
        t = round(time, 7)
        x = np.clip(state, -1, 1)
        u = np.clip(action, -1, 1)

        # Identify data_type
        if t <= self.tT - self.dt:
            data_type = 'path'
        else:
            data_type = 'terminal'

        # Environment option: Uncertain parameter?
        if len(args) == 0:  # Certain parameter
            p_mu, p_sigma, p_eps = self.param_real, np.zeros([self.p_dim, 1]), np.zeros([self.p_dim, 1])
        elif len(args) == 3:  # Uncertain parameter
            p_mu, p_sigma, p_eps = args

        # Integrate ODE
        if data_type == 'path':
            res = self.I_fnc(x0=x, p=u)
            xplus = res['xf'].full()
            tplus = t + self.dt
        else:
            xplus = x
            tplus = t

        # Compute output
        x_noise = np.zeros([self.s_dim, 1])
        y_noise = np.zeros([self.o_dim, 1])
        if self.prob_type == 'plant':
            x_noise[self.state_noise_idx] = self.state_noise * self.nprandom.standard_normal(size=[len(self.state_noise_idx), 1])
            y_noise[self.meas_noise_idx] = self.meas_noise * self.nprandom.standard_normal(size=[len(self.meas_noise_idx), 1])
        xplus = np.clip(xplus + x_noise, -1, 1)
        yplus = self.y_fnc(xplus, u, p_mu, p_sigma, p_eps).full()
        yplus = np.clip(yplus + y_noise, -1, 1)

        # Compute cost and constraint functions
        if data_type == 'path':
            cost = res['qf'].full()
            const = self.gP_fnc(x, u, p_mu, p_sigma, p_eps).full()
        elif data_type == 'terminal':
            cost = self.cT_fnc(x, p_mu, p_sigma, p_eps).full()
            const = self.gT_fnc(x, p_mu, p_sigma, p_eps).full()

        return tplus, xplus, yplus, u, cost, const, data_type

    def system_functions(self, *args):
        x, u, p_mu, p_sigma, p_eps = args

        x = self.descale(x, self.xmin, self.xmax)
        u = self.descale(u, self.umin, self.umax)

        x = ca.fmin(ca.fmax(x, self.xmin), self.xmax)
        u = ca.fmin(ca.fmax(u, self.umin), self.umax)

        t, x1, x2 = ca.vertsplit(x)
        u = ca.vertsplit(u)[0]

        dtdt = 1.
        dx1dt = x2
        dx2dt = -x1 + x2 * (1 - x2 ** 2) + u

        dx = [dtdt, dx1dt, dx2dt]

        dx = ca.vertcat(*dx)
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        outputs = ca.vertcat(t, x1, x2)
        y = self.scale(outputs, self.ymin, self.ymax, shift=True)

        return dx, y

    def cost_functions(self, data_type, *args):
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args  # scaled variable
        else:
            x, p_mu, p_sigma, p_eps = args # scaled variable
            u = np.zeros([self.a_dim, 1])

        Q = np.diag([0.0, 0.1, 0.1])
        R = np.diag([0.1])
        H = np.diag([0.0, 0.0, 0.0])

        uref = np.diag([0.0])

        y = self.y_fnc(x, u, p_mu, p_sigma, p_eps)

        if data_type == 'path':
            cost = y.T @ Q @ y + (u - uref).T @ R @ (u - uref)
        elif data_type == 'terminal': # terminal condition
            cost = y.T @ H @ y

        # Define Estimator cost
        self.Qe = 1E-4 * np.eye(self.o_dim)  # Path measurement error cost
        self.Re = np.eye(self.s_dim)  # Path state error cost
        self.He = 1E-4 * np.eye(self.s_dim)  # Arrival state error cost

        return cost

    def constraint_functions(self, data_type, *args):
        # Path Inequality constraints: cPineq_dim
        # Path Equality constraints: cPeq_dim
        # Terminal Inequality constraints: cTineq_dim
        # Terminal Equality constraints: cTeq_dim

        # scaled variable
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args
        elif data_type == 'leg':
            x, leg, p_mu, p_sigma, p_eps = args

        S = 5
        if data_type == 'path':
            x = self.descale(x, self.xmin, self.xmax)
            u = self.descale(u, self.umin, self.umax)
            t, x1, x2 = ca.vertsplit(x)
            g1 = u + x1 / 6
            g2 = -u - x1 / 6 - 1
            g = ca.vertcat(g1, g2) # Path inequality constraint
            g = self.scale(g, self.gmin, self.gmax, shift=False) * S
        elif data_type == 'terminal': # Terminal constraint --> u-included constraints are omitted
            g = np.array([[-10.]])
        return g

    def sym_expressions(self):
        """Syms: :Symbolic expressions, Fncs: Symbolic input/output structures"""

        # lists of sym_vars
        self.path_sym_args = [self.state_var, self.action_var, self.param_mu_var, self.param_sigma_var, self.param_epsilon_var]
        self.term_sym_args = [self.state_var, self.param_mu_var, self.param_sigma_var, self.param_epsilon_var]

        self.path_sym_args_str = ['x', 'u', 'p_mu', 'p_sig', 'p_eps']
        self.term_sym_args_str = ['x', 'p_mu', 'p_sig', 'p_eps']

        "Symbolic functions of f, y"
        self.f_sym, self.y_sym = self.system_functions(*self.path_sym_args)
        self.f_fnc = ca.Function('f_fnc', self.path_sym_args, [self.f_sym], self.path_sym_args_str, ['f'])
        self.y_fnc = ca.Function('y_fnc', self.path_sym_args, [self.y_sym], self.path_sym_args_str, ['y'])

        "Symbolic functions of c, cT, g"
        self.c_sym = partial(self.cost_functions, 'path')(*self.path_sym_args)
        self.cT_sym = partial(self.cost_functions, 'terminal')(*self.term_sym_args)
        self.gP_sym = partial(self.constraint_functions, 'path')(*self.path_sym_args)  # Path constraint
        self.gT_sym = partial(self.constraint_functions, 'terminal')(*self.term_sym_args)  # Terminal constraint

        self.c_fnc = ca.Function('c_fnc', self.path_sym_args, [self.c_sym], self.path_sym_args_str, ['c'])
        self.cT_fnc = ca.Function('cT_fnc', self.term_sym_args, [self.cT_sym], self.term_sym_args_str, ['cT'])
        self.gP_fnc = ca.Function('gP_fnc', self.path_sym_args, [self.gP_sym], self.path_sym_args_str, ['gP'])
        self.gT_fnc = ca.Function('gT_fnc', self.term_sym_args, [self.gT_sym], self.term_sym_args_str, ['gT'])

        "Symbolic function of dae solver"
        dae = {'x': self.state_var, 'p': self.action_var, 'ode': self.f_sym, 'quad': self.c_sym}
        opts = {'t0': 0., 'tf': self.dt}
        self.I_fnc = ca.integrator('I', 'cvodes', dae, opts)

    def eval_model_derivs(self):
        def jac_hess_eval(fnc, x_var, u_var):
            # Compute derivatives of cT, gT, gP
            fnc_dim = fnc.shape[0]

            dfdx = ca.jacobian(fnc, x_var)
            d2fdx2 = [ca.jacobian(dfdx[i, :], x_var) for i in range(fnc_dim)]

            if u_var is None:  # cT, gT
                if fnc_dim == 1:
                    dfdx = dfdx.T
                return [dfdx, *d2fdx2]
            else:  # gP, gL, gM
                dfdu = ca.jacobian(fnc, u_var)
                d2fdxu = [ca.jacobian(dfdx[i, :], u_var) for i in range(fnc_dim)]
                d2fdu2 = [ca.jacobian(dfdu[i, :], u_var) for i in range(fnc_dim)]
                if fnc_dim == 1:
                    dfdx = dfdx.T
                    dfdu = dfdu.T
                return [dfdx, dfdu, *d2fdx2, *d2fdxu, *d2fdu2]

        f_derivs = ca.Function('f_derivs', self.path_sym_args,
                               [self.f_sym] + jac_hess_eval(self.f_sym, self.state_var,
                                                            self.action_var))  # ["F", "Fx", "Fu", "Fxx", "Fxu", "Fuu"]
        c_derivs = ca.Function('c_derivs', self.path_sym_args,
                               [self.c_sym] + jac_hess_eval(self.c_sym, self.state_var,
                                                            self.action_var))  # ["L", "Lx", "Lu", "Lxx", "Lxu", "Luu"]
        cT_derivs = ca.Function('cT_derivs', self.term_sym_args,
                                [self.cT_sym] + jac_hess_eval(self.cT_sym, self.state_var,
                                                              None))  # ["LT", "LTx", "LTxx"]
        gP_derivs = ca.Function('gP_derivs', self.path_sym_args,
                                [self.gP_sym] + jac_hess_eval(self.gP_sym, self.state_var,
                                                              self.action_var))  # ["GP", "GPx", "GPu", "GPxx", "GPxu", "GPuu"]
        gT_derivs = ca.Function('gT_derivs', self.term_sym_args,
                                [self.gT_sym] + jac_hess_eval(self.gT_sym, self.state_var,
                                                              None))  # ["GT", "GTx", "GTxx"]

        return f_derivs, c_derivs, cT_derivs, gP_derivs, gT_derivs

    def initial_control(self, x):
        u = np.array([[0., -0.5, -0.5]]) @ x
        return u

    def scale(self, var, min, max, shift=True):
        if self.zero_center_scale == True:  # [min, max] --> [-1, 1]
            shifting_factor = max + min if shift else 0.
            scaled_var = (2. * var - shifting_factor) / (max - min)
        else:  # [min, max] --> [0, 1]
            shifting_factor = min if shift else 0.
            scaled_var = (var - shifting_factor) / (max - min)

        return scaled_var

    def descale(self, scaled_var, min, max):
        if self.zero_center_scale == True:  # [-1, 1] --> [min, max]
            var = (max - min) / 2 * scaled_var + (max + min) / 2
        else:  # [0, 1] --> [min, max]
            var = (max - min) * scaled_var + min
        return var
