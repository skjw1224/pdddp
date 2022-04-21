import numpy as np
from os import path
import casadi as ca
from functools import partial

from numpy.random import Generator, PCG64

class CrystalEnv(object):
    # Certain paramaters
    Cs = 0.46
    g = 1
    Gmax = 5E-8
    Hc = 60.75
    Hl = 69.86
    Hv = 2.59E3
    Kv = 0.43
    kb = 1.02E15
    kg = 7.5E-5
    Fp = 1.73E-6
    V = 7.5E-2
    rhoc = 1767.35
    rhol = 1248.93

    def __init__(self, prob_type=None):
        self.name = "Crystallization"
        self.prob_type = prob_type

        # Dimensions and variables
        self.s_dim = 8
        self.a_dim = 1
        self.o_dim = 8

        # Initial guess of uncertain parameters
        self.param_real = np.array([[]]).T
        self.param_mu_prior = np.array([[]]).T
        self.param_sigma_prior = np.array([[]]).T
        self.p_dim = len(self.param_real)

        # Four types of constraints
        self.cPineq_dim, self.cPeq_dim = 3, 0 # Path constraint
        self.cTineq_dim, self.cTeq_dim = 3, 0  # Global terminal constraint

        self.c_dim = [self.cPineq_dim, self.cPeq_dim, self.cTineq_dim, self.cTeq_dim]

        # MX variable for dae function object (no SX)
        self.state_var = ca.MX.sym('x', self.s_dim)
        self.action_var = ca.MX.sym('u', self.a_dim)
        self.param_mu_var = ca.MX.sym('p_mu', self.p_dim)
        self.param_sigma_var = ca.MX.sym('p_sig', self.p_dim)
        self.param_epsilon_var = ca.MX.sym('p_eps', self.p_dim)
        self.ref_var = None

        self.real_env = False

        # m0i, m1i, m2i, m3i, m4i = self.crystal_size_initial_dist()
        m0i, m1i, m2i, m3i, m4i = [1E11, 4E6, 400, 0.1, 0.2E-4]
        C0 = 0.461
        Q0 = 9. #kW

        self.x0 = np.array([[0., m0i, m1i, m2i, m3i, m4i, C0, Q0]]).T
        self.u0 = np.array([[0.]]).T
        self.t0 = 0.
        self.dt = 10 # s
        self.tT = 10000
        self.nT = int(self.tT / self.dt) + 1
        self.eps_t = np.finfo(np.float32).eps

        self.xmin = np.array([[self.t0, m0i, m1i, m2i, m3i, m4i, self.Cs, 7]]).T
        self.xmax = np.array([[self.tT, m0i * 1.2, m1i * 10, m2i * 10, m3i *10, m4i*10, C0, 20]]).T
        self.ymin = self.xmin
        self.ymax = self.xmax
        self.umin = np.array([[-0.02]]).T
        self.umax = np.array([[0.02]]).T
        self.gmin = np.array([[0., 8, 8]]).T
        self.gmax = np.array([[self.Gmax, 20, 20]]).T

        self.cmin = np.array([[m3i / m2i - (m3i / m2i) / 2, 7]]).T
        self.cmax = np.array([[m3i / m2i, 20]]).T

        # Define noise related values
        self.init_state_idx = []
        self.state_noise_idx = [1, 2, 3, 4, 5, 6]
        self.meas_noise_idx = []
        self.init_state_noise = 0.02
        self.state_noise = 0.005
        self.meas_noise = 0.04
        self.nprandom = Generator(PCG64(2021))

        self.zero_center_scale = False

        self.sym_expressions()
        self.model_derivs = self.eval_model_derivs()

        self.reset()

    def reset(self, x0=None):

        if x0 is None:
            x0 = self.x0

        state = self.scale(x0, self.xmin, self.xmax)
        if self.prob_type == 'plant':
            # States with random initial values
            state[self.init_state_idx] = \
                state[self.init_state_idx] + self.nprandom.uniform(-self.init_state_noise, self.init_state_noise,
                                                                   [len(self.init_state_idx), 1])
            state = np.clip(state, 0, 1)

        time = self.t0

        p_mu, p_sigma, p_eps = self.param_real, np.zeros([self.p_dim, 1]), np.zeros([self.p_dim, 1])
        y = self.y_fnc(state, self.u0, p_mu, p_sigma, p_eps).full()
        return time, state, y

    def step(self, time, state, action, *args):
        # Scaled state, action, output
        t = round(time, 7)
        x = np.clip(state, 0, None)
        u = action

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
        xplus = np.clip(xplus + x_noise, 0, None)
        yplus = self.y_fnc(xplus, u, p_mu, p_sigma, p_eps).full()
        yplus = np.clip(yplus + y_noise, 0, None)

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

        x = ca.fmax(x, self.xmin)
        u = ca.fmin(ca.fmax(u, self.umin), self.umax)

        Cs, g, Hc, Hl, Hv, Kv, kb, kg, Fp, V, rhoc, rhol = self.Cs, self.g, self.Hc, self.Hl, self.Hv, \
                                                           self.Kv, self.kb, self.kg, self.Fp, self.V, self.rhoc, self.rhol

        t, m0, m1, m2, m3, m4, C, Q = ca.vertsplit(x)
        dQ = ca.vertsplit(u)[0]

        k1 = Hv * Cs / (Hv - Hl) * (rhoc / rhol - 1 + (rhol * Hl - rhoc * Hc) / (rhol * Hv)) - rhoc / rhol
        k2 = Cs / (V * rhol * (Hv - Hl))
        G = kg * (C - Cs) ** g
        B0 = kb * m3 * G

        dtdt = 1.
        dm0dt = B0 - m0 * Fp / V
        dm1dt = G * m0 - m1 * Fp / V
        dm2dt = 2 * G * m1 - m2 * Fp / V
        dm3dt = 3 * G * m2 - m3 * Fp / V
        dm4dt = 4 * G * m3 - m4 * Fp / V
        dCdt = (Fp * (Cs - C) / V + 3 * Kv * G * m2 * (k1 + C)) / (1 - Kv * m3) + k2 * Q / (1 - Kv * m3)
        dQdt = dQ

        dx = [dtdt, dm0dt, dm1dt, dm2dt, dm3dt, dm4dt, dCdt, dQdt]

        dx = ca.vertcat(*dx)
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        outputs = x
        y = self.scale(outputs, self.ymin, self.ymax, shift=True)

        return dx, y

    def cost_functions(self, data_type, *args):
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args  # scaled variable
        else:
            x, p_mu, p_sigma, p_eps = args # scaled variable
            u = np.zeros([self.a_dim, 1])

        Q = np.diag([1.]) * 0.0001
        R = np.diag([1.]) * 0.0001
        H = np.diag([1.]) * 10
        uref = np.diag([0.5])

        x = self.descale(x, self.xmin, self.xmax)
        t, m0, m1, m2, m3, m4, C, Qv = ca.vertsplit(x)
        cost = ca.vertcat(m3 / m2, Qv)
        d32, Qv = ca.vertsplit(self.scale(cost, self.cmin, self.cmax, shift=True))

        if data_type == 'path':
             # Delu --> 0.5 in [0, 1] scale
            cost = - d32 @ Q + Qv ** 2 * Q + (u - uref).T @ R @ (u - uref)
        elif data_type == 'terminal': # terminal condition
            cost = - d32 @ H

        return cost

    def constraint_functions(self, data_type, *args):
        # Path Inequality constraints: cPineq_dim
        # Path Equality constraints: cPeq_dim
        # Terminal Inequality constraints: cTineq_dim
        # Terminal Equality constraints: cTeq_dim

        # scaled variable
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args
        elif data_type == 'terminal':
            x, p_mu, p_sigma, p_eps = args

        S = 1 # 0.001
        T = 1 #0.001

        if data_type == 'path':
            x = self.descale(x, self.xmin, self.xmax)
            t, m0, m1, m2, m3, m4, C, Qv = ca.vertsplit(x)
            g1 = ca.if_else(t - 100, (self.kg * (C - self.Cs) ** self.g - self.Gmax), -1)
            g2 = Qv - 13
            g3 = -Qv + 9
            g = ca.vertcat(g1, g2, g3) # Path inequality constraint
            g = self.scale(g, self.gmin, self.gmax, shift=False)
            g = g*S

        elif data_type == 'terminal': # Terminal constraint --> u-included constraints are omitted
            xs = self.descale(x, self.xmin, self.xmax)
            t, m0, m1, m2, m3, m4, C, Qv = ca.vertsplit(xs)
            g1 = ca.if_else(t - 100, (self.kg * (C - self.Cs) ** self.g - self.Gmax), -1)
            g2 = Qv - 13
            g3 = -Qv + 9
            g = ca.vertcat(g1, g2, g3)
            g = self.scale(g, self.gmin, self.gmax, shift=False)
            g = g * T
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
        dQ = 0.5 * np.ones([self.a_dim, 1])
        return dQ

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

    def crystal_size_initial_dist(self, ngrid=101):
        """J. M. Schall et al., Cryst Growth Des. 2018, 18, 1560-1570"""
        Lm = 0.01E-3
        LM = 1.2E-3
        M = 2.8E15
        sig = 0.95
        dL = (LM - Lm)/(ngrid - 1)
        x = np.linspace(Lm, LM, ngrid)
        y = self.kb / self.kg * M * sig * x ** 4 * np.exp(-x / (self.kg * sig ** 2))

        m0i = np.sum(y) * dL
        m1i = np.dot(x, y) * dL
        m2i = np.dot(x**2, y) * dL
        m3i = np.dot(x**3, y) * dL
        m4i = np.dot(x**4, y)* dL

        return [m0i, m1i, m2i, m3i, m4i]
