import numpy as np
from pdddp.hyperparameter import HyperParameter

HYP_PAR_DIM = int(3)
LAGR_MAX = 10
X_MAX = 1
U_MAX = 1

class DynamicsIterator(object):
    def __init__(self, env, noise_training_idx, epi_max):
        self.env = env
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.o_dim = env.o_dim
        self.c_dim_list = env.c_dim
        self.cPineq_dim, self.cPeq_dim, self.cTineq_dim, self.cTeq_dim = self.c_dim_list
        self.cP_dim = self.cPineq_dim + self.cPeq_dim
        self.cT_dim = self.cTineq_dim + self.cTeq_dim

        self.zero_center_scale = env.zero_center_scale

        # Nonnegative Lagrangian for inequality constraints, equality constraints --> LB: -LAGRMAX
        self.LAGR_MIN_Pineq_masked = np.concatenate([np.zeros([self.cPineq_dim, 1]), -LAGR_MAX * np.ones([self.cPeq_dim, 1])])
        self.LAGR_MIN_Tineq_masked = np.concatenate([np.zeros([self.cTineq_dim, 1]), -LAGR_MAX * np.ones([self.cTeq_dim, 1])])

        self.X_MAX, self.U_MAX = X_MAX, U_MAX
        if self.zero_center_scale == True:
            self.X_MIN, self.U_MIN = -X_MAX, -U_MAX
        else:
            self.X_MIN, self.U_MIN = 0, 0

        self.t0 = env.t0  # ex) 0
        self.tT = env.tT  # ex) 2
        self.nT = env.nT  # ex) dt:0.005 nT = 401
        self.tvec = np.linspace(self.t0, self.tT, num=self.nT) # Equidistant split

        # Equidistant split
        self.leg_idx = [0, self.nT - 1]
        self.noise_training_idx = noise_training_idx

        # Initialize hyperparameters
        self.hyper_parameter = HyperParameter(self.s_dim, self.a_dim, self.c_dim_list, epi_max)
        self.hyper_param_path = self.hyper_parameter.path_hyperparam_init()
        self.hyper_param_term = self.hyper_parameter.boundary_hyperparam_init()

    def multiple_leg_rollout(self, epi_solutions, prev_epi_data, controller, estimator):
        """Multiple leg rollout data generation. Use schedule of variables: path(u, l), terminal(muT)
        and hyperparams(path, master, leg, terminal)

        Output: data structure & miscellaneous data structure
        path data: x, x2, u, l, [c, tr, tol], pm, ps, pe,  -------------   path misc data: rT, yT, path const
        term data: xT, mT, [c, tr, tol], pm, ps, pe             -------------   term misc data: rT, yT, term const
        """

        if epi_solutions is not None: # Intermediate solutions
            epi_path_solution, epi_term_solution = epi_solutions
        else: # Initial control
            epi_path_solution, epi_term_solution = None, None

        if prev_epi_data is not None: # Test --> Use episode data from the final iteration of training phase to get the nominal state trajectory
            prev_epi_path_data, prev_epi_term_data = prev_epi_data
        else: # Training --> Do not use previous episode data
            prev_epi_path_data, prev_epi_term_data = None, None


        epi_path_data, epi_term_data = [], []
        epi_path_misc_data, epi_term_misc_data = [], []

        # Controller: 'Initial', 'MPC', 'Open-loop DDP', 'Corrector DDP', 'Predictor-corrector DDP'
        self.controller_type = list(controller.keys())[0]
        self.controller = controller[self.controller_type]
        self.estimator = estimator[list(estimator.keys())[0]]


        # Collocate
        t0, x0, y0 = self.env.reset()
        t, x, y = t0, x0, y0
        x_est = y

        for i in range(self.leg_idx[0], self.leg_idx[1]):
            # Path schedule
            u, l, hyppar, p_mu, p_sigma, p_eps, epi_path_solution \
                = self.path_ctrl_schedule(i, x_est, epi_path_solution, prev_epi_path_data)

            # Path rollout
            t2, x2, y2, u, r, const, data_type = self.env.step(t, x, u)

            x2_est = self.path_est_schedule(y2)

            # Path data storage
            epi_path_data.append([x, x_est, u, l, hyppar, p_mu, p_sigma, p_eps])
            epi_path_misc_data.append([r, y2, const])

            # Proceed loop
            t = t2
            x = x2
            x_est = x2_est

        # Boundary rollout
        tT, xT, yT, uT, rT, constT, data_type = self.env.step(t, x, u)

        # Data store
        xT_est = self.path_est_schedule(y2)
        # Final leg --> Terminal rollout
        muT, hyppar_term, p_mu, p_sigma, p_eps = self.boundary_schedule_at_term(xT_est, epi_term_solution, prev_epi_term_data)

        epi_term_data.append([xT, xT_est, muT, hyppar_term, p_mu, p_sigma, p_eps])
        epi_term_misc_data.append([rT, yT, constT])


        epi_data = [epi_path_data, epi_term_data]
        epi_misc_data = [epi_path_misc_data, epi_term_misc_data]
        epi_solutions = [epi_path_solution, epi_term_solution]

        return epi_data, epi_misc_data, epi_solutions

    def path_ctrl_schedule(self, i, x, epi_path_solution, epi_path_data):
        if self.controller_type == 'Initial' or self.controller_type == 'MPC':
            u = self.controller(x)
            lagr = np.zeros([self.cP_dim, 1])
            hyppar = self.hyper_param_path
        else: # Controller: DDP
            V_epi, path_gain_epi, u_epi, l_epi, hypparam_epi, _, _ = epi_path_solution
            u = u_epi[i]
            lagr = l_epi[i]
            hyppar = hypparam_epi[i]

            if self.controller_type != 'Open-loop DDP': # Controller: Corrector or Predictor-Corrector DDP
                x_nom = epi_path_data[i][0]
                delx = x - x_nom

                l, Kx = path_gain_epi[i]  # [lu, ll]; [Kux, Klx];

                path_gain_epi[i] = [l, Kx]
                delul = l + Kx @ delx
                delu, dell = delul[:self.a_dim, :], delul[self.a_dim:, :]
                u = u + delu
                lagr = lagr + dell

                if self.controller_type == 'Predictor-corrector DDP':
                    # Update x, u, step size

                    epi_path_data[i][1] = x

                    epi_path_data[i][2] = u
                    epi_path_data[i][4][1] = 0.01
                    _, _, new_path_gain_epi, _, Hul_norm, _, new_data = \
                    self.controller.bwd_path_eval(env_data=epi_path_data[i], new_Vnplus_list=V_epi[i], prev_b_moments=[0, 0])

                    l, Kx = new_path_gain_epi

                    delul = l
                    delu, dell = delul[:self.a_dim, :], delul[self.a_dim:, :]
                    u = u + delu
                    lagr = lagr + dell

                    epi_path_solution[1][i] = new_path_gain_epi


        u = np.clip(u, self.U_MIN, self.U_MAX)
        lagr = np.clip(lagr, self.LAGR_MIN_Pineq_masked, LAGR_MAX)
        p_mu, p_sigma, p_eps = [], [], []

        return u, lagr, hyppar, p_mu, p_sigma, p_eps, epi_path_solution

    def boundary_schedule_at_term(self, xT, epi_solution, epi_data):
        if epi_solution is None: # Initial control
            # Assign zero values for initial policy
            mu = np.zeros([self.cT_dim, 1])
            hyppar = self.hyper_param_term

        else:
            _,  term_gain_epi, muT_epi, hypparam_term_epi, _ = epi_solution
            if epi_data is None:
                mu = muT_epi[0]
            else:
                # Test phase
                # xT: plant-data --> closed-loop solution need to be computed here
                xT_nom = epi_data[0][0]
                lm, Kmx = term_gain_epi[0]
                delxT = xT - xT_nom
                delm = lm + Kmx @ delxT
                mu = muT_epi[0] + delm

            if hypparam_term_epi == None:
                hyppar = self.hyper_param_term
            else:
                hyppar = hypparam_term_epi[0]

        p_mu, p_sigma, p_eps = [], [], []

        return mu, hyppar, p_mu, p_sigma, p_eps

    def path_est_schedule(self, y):
        if self.estimator is not None:
            x_est = self.estimator(y)
        else:
            x_est = y
        return x_est
