
import numpy as np
import do_mpc

MAX_ITER = 1000
ne, nd = 1, 3
nP, nR = 10, 5
M = 10000

class DirectCollocation(object):
    def __init__(self, env, track_ref=None, full_horizon=True, closed_loop=False, mpc_period=None):
        self.env = env
        self.track_ref = track_ref
        self.full_horizon = full_horizon
        self.closed_loop = closed_loop
        self.mpc_period = mpc_period
        self.t_idx = 0
        self.mhe_idx = 0
        self.mpc_called = False

        # self.sd_dim = self.env.sd_dim  # differential state
        # self.sa_dim = self.env.sa_dim  # algebraic state
        self.sd_dim = self.env.s_dim
        self.s_dim = self.env.s_dim  # Diff + Alg states
        self.o_dim = self.env.o_dim
        self.a_dim = self.env.a_dim
        self.p_dim = self.env.p_dim

        self.c_dim_list = env.c_dim
        self.cPineq_dim, self.cPeq_dim, self.cTineq_dim, self.cTeq_dim = self.c_dim_list
        self.cP_dim = self.cPineq_dim + self.cPeq_dim
        self.cT_dim = self.cTineq_dim + self.cTeq_dim

        self.t0 = self.env.t0  # Initial time
        self.dt = self.env.dt  # Sampling time
        self.tT = self.env.tT  # Time horizon
        self.nT = self.env.nT  # number of control intervals

        self.ne = ne  # Number of elements within one dt (sampling time)
        self.nd = nd  # Degree of interpolating polynomial
        self.nR = nR  # Retrodiction horizon for MHE

        # Prediction horizon for MPC
        if self.full_horizon:
            self.nP = self.nT - 1
        else:
            self.nP = nP

        self.abstol = 1e-10
        self.reltol = 1e-10

        # Operating region
        self.ub = 1.
        if self.env.zero_center_scale is True:
            self.lb = -1.
        else:
            self.lb = 0.

        self.f_fnc = self.env.f_fnc
        self.y_fnc = self.env.y_fnc
        self.c_fnc = self.env.c_fnc
        self.cT_fnc = self.env.cT_fnc

        self.p_mu = self.env.param_real
        self.p_sigma = np.zeros([self.p_dim, 1])
        self.p_eps = np.zeros([self.p_dim, 1])

        # Sc = np.diag(np.ones(s_dim)) * 0.1
        Sc = np.array([[2.]]).T
        # slack_mask = np.array([[1, 1, 1, 0, 0]]).T
        slack_mask = np.array([[0, 0, 0, 0, 0]]).T

        # Define dompc objects
        self.set_dompc_model()
        self.set_dompc_simulator()
        # self.set_dompc_estimator()

    def set_dompc_model(self):
        # Set do-mpc model
        self.dompc_model = do_mpc.model.Model('continuous')  # 'discrete' or 'continuous'

        # Variables and equations from model class
        x_t = self.env.state_var
        u_t = self.env.action_var
        pm_t = self.env.param_mu_var
        ps_t = self.env.param_sigma_var
        pe_t = self.env.param_epsilon_var
        r_t = self.env.ref_var

        # x, u, p struct (optimization variables):
        x = self.dompc_model.set_variable(var_type='_x', var_name='x', shape=x_t.shape)
        u = self.dompc_model.set_variable(var_type='_u', var_name='u', shape=u_t.shape)
        pm = self.dompc_model.set_variable(var_type='_p', var_name='pm', shape=pm_t.shape)
        ps = self.dompc_model.set_variable(var_type='_p', var_name='ps', shape=ps_t.shape)
        pe = self.dompc_model.set_variable(var_type='_p', var_name='pe', shape=pe_t.shape)

        if r_t is not None:
            ref = self.dompc_model.set_variable(var_type='_tvp', var_name='ref', shape=r_t.shape)
        else:
            ref = None

        # Measurements
        x_meas = self.dompc_model.set_meas('x_meas', x, meas_noise=True)

        # Differential equations
        dx_eval = self.env.f_fnc(x, u, pm, ps, pe)
        self.dompc_model.set_rhs('x', dx_eval, process_noise=False)

        aux = self.env.gP_fnc(x, u, pm, ps, pe)
        for i in range(aux.shape[0]):
            self.dompc_model.set_expression('aux_' + str(i), aux[i])

        # Build the model
        self.dompc_model.setup()

    def set_dompc_simulator(self):
        self.dompc_simulator = do_mpc.simulator.Simulator(self.dompc_model)

        params_simulator = {
            'integration_tool': 'idas',
            'abstol': self.abstol,
            'reltol': self.reltol,
            't_step': self.dt
        }

        # Set parameter(s):
        self.dompc_simulator.set_param(**params_simulator)

        # Set function for parameters
        p_struct = self.dompc_simulator.get_p_template()

        p_keys = p_struct.keys()[1:]  # First key is 'default'
        tips = [self.p_mu, self.p_sigma, self.p_eps]

        def p_fnc(t_now):
            for i, pi in enumerate(tips):
                p_struct[p_keys[i]] = pi

            return p_struct

        self.dompc_simulator.set_p_fun(p_fnc)

        # Set tvp
        tvp_struct = self.dompc_simulator.get_tvp_template()
        tvp_keys = tvp_struct.keys()[1:] # First element is 'default'
        def tvp_fnc_wrapper(t_now):
            for key in tvp_keys:
                tvp_struct[key] = self.get_tvp(t_now)

            return tvp_struct

        self.dompc_simulator.set_tvp_fun(tvp_fnc_wrapper)

        # Setup simulator:
        self.dompc_simulator.setup()

    def set_dompc_mpc(self):
        model = self.dompc_model
        simulator = self.dompc_simulator
        mpc = do_mpc.controller.MPC(model)

        # https://www.do-mpc.com/en/latest/api/do_mpc.controller.MPC.set_param.html
        setup_mpc = {
            'n_horizon': self.nP,
            'n_robust': 0,
            't_step': self.dt,
            # 'collocation_type ': 'legendre',
            'collocation_deg': self.nd,  # Degree of interpolating polynomial
            'collocation_ni': self.ne,  # Number of elements within one dt (sampling time)
            'store_full_solution': True,
            'nl_cons_check_colloc_points': True,
            # 'store_solver_stats': ['success','t_wall_S','iterations'],
            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {
                # 'ipopt.linear_solver': 'ma27',
                'ipopt.tol': 1E-6,
                'ipopt.max_iter': 500,
                # 'ipopt.print_level': 0,
                # 'print_time': 0,
                # 'ipopt.sb': 'yes'
            }
        }

        if self.closed_loop:
            setup_mpc['nlpsol_opts'].update({'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes'})


        mpc.set_param(**setup_mpc)

        # Time-invariant parameters
        p_template_mpc = simulator.get_p_template()

        p_keys = p_template_mpc.keys()[1:]  # First key is 'default'
        p_vals = simulator.p_fun(0.0)  # Get true parameter from simulator
        p_vals = np.array(p_vals.cat)  # DM is not iteratl
        p_mpc = dict(zip(p_keys, p_vals))

        if p_vals.shape[0] > 0:
            mpc.set_uncertainty_values(**p_mpc)

        # Time-varying parameters
        tvp_struct = mpc.get_tvp_template()
        tvp_keys = tvp_struct.keys()[1:]  # Firs
        # Encode time-varying parameters
        def tvp_fun_wrapper(t_now):
            t = t_now
            for i in range(mpc.n_horizon + 1):
                tvp_struct['_tvp', i] = self.get_tvp(t)
                t = t + mpc.t_step
            return tvp_struct

        mpc.set_tvp_fun(tvp_fun_wrapper)


        if self.env.ref_var is not None:
            cost_args = [model.x, model.u, model.p['pm'], model.p['ps'], model.p['pe'], model.p['ref']]
            costT_args = [model.x, model.p['pm'], model.p['ps'], model.p['pe'], model.p['ref']]
        else:
            cost_args = [model.x, model.u, model.p['pm'], model.p['ps'], model.p['pe']]
            costT_args = [model.x, model.p['pm'], model.p['ps'], model.p['pe']]
        self.lterm = self.env.c_fnc(*cost_args) * self.env.dt
        self.mterm = self.env.cT_fnc(*costT_args)

        mpc.set_objective(mterm=self.mterm, lterm=self.lterm)

        # mpc.set_rterm(Flow_ctrl=R)

        # Scalings, bounds for x, z, u
        for i, xstri in enumerate(model.x.keys()):
            mpc.scaling['_x', xstri] = 1.
            mpc.bounds['lower', '_x', xstri] = self.lb
            mpc.bounds['upper', '_x', xstri] = self.ub

        for i, ustri in enumerate(model.u.keys()[1:]):  # first key is 'default'
            mpc.scaling['_u', ustri] = 1.
            mpc.bounds['lower', '_u', ustri] = self.lb
            mpc.bounds['upper', '_u', ustri] = self.ub

        for i, auxstri in enumerate(model.aux.keys()[1:]):
            mpc.set_nl_cons('cons_' + str(i), model.aux[auxstri], ub=0., soft_constraint=True, penalty_term_cons=100.)

        mpc.setup()
        self.dompc_mpc = mpc

    def set_dompc_estimator(self):
        model = self.dompc_model
        # simulator = self.dompc_simulator
        # mhe = do_mpc.estimator.MHE(model)

        # # Set parameters:
        # setup_mhe = {
        #     'n_horizon': self.nR,
        #     't_step': self.dt,
        #     'collocation_deg': self.nd,
        #     'collocation_ni': self.ne,
        #     'store_full_solution': True,
        #     'meas_from_data': True,
        #     'nlpsol_opts': {
        #         # 'ipopt.linear_solver': 'ma27',
        #         'ipopt.tol': 1E-6,
        #         'ipopt.max_iter': 1000,
        #         'ipopt.print_level': 1,
        #         'print_time': 0,
        #         'ipopt.sb': 'yes'
        #     }
        # }
        #
        # mhe.set_param(**setup_mhe)
        #
        # # Problem size (possibly truncated)
        # n_steps = min(mhe.data._y.shape[0], mhe.n_horizon)
        #
        # # Time-invariant parameters
        # p_template_mhe = simulator.get_p_template()
        # p_keys = p_template_mhe.keys()[1:]  # First key is 'default'
        # tips = [self.p_mu, self.p_sigma, self.p_eps]
        #
        # def p_fun(t_now):
        #
        #     p_vals = simulator.p_fun(0.0)  # Get true parameter from simulator
        #     p_vals = np.array(p_vals.cat)  # DM is not iteratl
        #
        #     for i, key in enumerate(p_keys):
        #         p_template_mhe[key] = tips[i]
        #
        #     return p_template_mhe
        #
        # mhe.set_p_fun(p_fun)
        #
        # # Set tvp
        # p_tvp = mhe.get_tvp_template()
        #
        # # Encode time-varying parameters
        # def tvp_fun_wrapper(t_now):
        #     t = t_now
        #     for i in range(-n_steps, 0):
        #         p_tvp['_tvp', i] = self.get_tvp(t)
        #         t = t - mhe.t_step
        #
        #     try:
        #         for i in range(mhe.n_hoziron - n_steps):
        #             p_tvp['_tvp', i] = self.get_tvp(t)
        #     except:
        #         None
        #
        #     return p_tvp
        #
        # mhe.set_tvp_fun(tvp_fun_wrapper)
        #
        # # Cost function
        # mhe.set_default_objective(P_x=self.env.He, P_v=self.env.Qe)
        #
        # # Scalings, bounds for x, z, u, p
        # for i, xstri in enumerate(model.x.keys()):
        #     mhe.scaling['_x', xstri] = 1.
        #     mhe.bounds['lower', '_x', xstri] = self.lb
        #     mhe.bounds['upper', '_x', xstri] = self.ub
        #
        # for i, ustri in enumerate(model.u.keys()[1:]):  # first key is 'default'
        #     mhe.scaling['_u', ustri] = 1.
        #     mhe.bounds['lower', '_u', ustri] = self.lb
        #     mhe.bounds['upper', '_u', ustri] = self.ub
        #
        # # for i, pstri in enumerate(mhe.p_est0.keys()[1:]):  # first key is 'default'
        # #     mhe.scaling['_p_est', pstri] = env.pmax[i]
        # #     mhe.bounds['lower', '_p_est', pstri] = env.pmin[i]
        # #     mhe.bounds['upper', '_p_est', pstri] = env.pmax[i]
        #
        # # for i in range(self.s_dim):
        # #     if i not in self.env.state_noise_idx:
        # #         mhe.set_nl_cons('state_cons_lb_' + str(i), model.w['x'][i],
        # #                         ub=0., soft_constraint=True, penalty_term_cons=100.)
        # #         mhe.set_nl_cons('state_cons_ub_' + str(i), -model.w['x'][i],
        # #                         ub=0., soft_constraint=True, penalty_term_cons=100.)
        #
        # for i in range(self.o_dim):
        #     if i not in self.env.meas_noise_idx:
        #         mhe.set_nl_cons('meas_cons_lb_' + str(i), model.v['x_meas'][i],
        #                         ub=0., soft_constraint=True, penalty_term_cons=100.)
        #         mhe.set_nl_cons('meas_cons_ub_' + str(i), -model.v['x_meas'][i],
        #                         ub=0., soft_constraint=True, penalty_term_cons=100.)
        #
        # # Setup
        # mhe.setup()
        self.dompc_estimator = do_mpc.estimator.StateFeedback(model)

    def get_tvp(self, t):
        if self.dompc_model.n_tvp > 0:
            t_index = min(int(t / self.dt), self.nT - 2)
            # get x ref value
            y_ref = self.track_ref[0][t_index][0]
            u_ref = self.track_ref[0][t_index][2]
            ref = np.concatenate([y_ref, u_ref])
        else:
            ref = []
        return ref

    def control(self, x0):
        # Case1: Full horizon, closed-loop, economic
        # Case2: Full horizon, open-loop, economic
        # Case3: Short horizon, closed-loop, economic

        self.dompc_simulator.x0 = x0

        if self.closed_loop:
            if self.full_horizon:
                # Shrinking horizon: reset MPC every time step
                self.nP = self.nT - self.t_idx - 1

                if self.mpc_period is None:
                    self.set_dompc_mpc()
                else:
                    if int(self.t_idx % self.mpc_period) == 0:
                        self.set_dompc_mpc()
            else:
                # Fixed horizon: define MPC once
                self.nP = nP
                if not self.mpc_called:
                    self.set_dompc_mpc()

            self.dompc_mpc.x0 = x0
            self.dompc_mpc.set_initial_guess()


            if self.mpc_period is None:
                u = self.dompc_mpc.make_step(x0)
            else:
                if int(self.t_idx % self.mpc_period) == 0:
                    u = self.dompc_mpc.make_step(x0)
                else:
                    u = np.array(self.dompc_mpc.opt_x_num_unscaled['_u', int(self.t_idx % self.mpc_period), 0])
        else:
            if not self.mpc_called:
                if self.full_horizon:
                    self.nP = self.nT- self.t_idx - 1
                else:
                    self.nP = nP
                self.set_dompc_mpc()
                self.dompc_mpc.x0 = x0
                self.dompc_mpc.set_initial_guess()
                # Open loop: compute MPC once
                u = self.dompc_mpc.make_step(x0)
            else:
                u = np.array(self.dompc_mpc.opt_x_num_unscaled['_u', self.t_idx, 0])

        self.t_idx += 1
        self.mpc_called = True
        return u

    def estimate(self, y):

        # self.dompc_estimator.x0 = x0
        # self.dompc_estimator.set_initial_guess()
        # x = self.dompc_estimator.make_step(y)
        x = y

        return x
