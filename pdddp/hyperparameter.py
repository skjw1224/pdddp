import numpy as np

# PEN_INIT = 1.
# PEN_INC_RATE, PEN_DEAC_RATE, PEN_MIN, PEN_MAX = 0.00, 0.00, 0.1, 1E2
# TR_MIN, TR_MAX = 1E-7, 0.1
# BETA, TOL_MIN, TOL_MAX = 0.2, 1E-8, 1E-2
# ALPHA, ALPHA_FIN, BETA1, BETA2, EPS = 0.01, 0.05, 0.8, 0.9, 1e-7  # Adam parameters beta1 < sqrt(beta2)

PEN_INIT = 1.
PEN_INC_RATE, PEN_DEAC_RATE, PEN_MIN, PEN_MAX = 0.001, 0.00, 0.1, 1E2
TR_MIN, TR_MAX = 1E-7, 0.1
BETA, TOL_MIN, TOL_MAX = 0.2, 1E-8, 1E-2
ALPHA, ALPHA_FIN, BETA1, BETA2, EPS = 0.01, 0.05, 0.8, 0.9, 1e-7  # Adam parameters beta1 < sqrt(beta2)

class HyperParameter(object):
    def __init__(self, s_dim, a_dim, c_dim_list, epi_max):
        self.s_dim, self.a_dim = s_dim, a_dim
        self.cPineq_dim, self.cPeq_dim, self.cTineq_dim, self.cTeq_dim = c_dim_list
        self.cP_dim = self.cPineq_dim + self.cPeq_dim
        self.cT_dim = self.cTineq_dim + self.cTeq_dim
        self.epi_max = epi_max

    def path_hyperparam_init(self):
        # L, Lx, Lu, _, _, _ = cost_fcns_eval_list
        # G, Gx, Gu, _, _, _ = const_fcns_eval_list
        # gradL_max = np.max(np.clip(np.abs(np.c_[Lx, Lu]), None, 5), axis=1)
        # gradG_max = np.max(np.clip(np.abs(np.c_[Gx, Gu]), None, 5), axis=1)
        # Opt = np.sum(np.abs(L / gradL_max))
        # Feas = np.sum(np.square(G / gradG_max)) * 0.5

        # penalty = np.clip(10 * min(1e-5, Opt) / min(1e-5, Feas), PEN_MIN, PEN_MAX)
        penalty = PEN_INIT
        tr_rad_path = ALPHA
        tol_path = np.clip((1 / (2 * penalty)) ** BETA * TOL_MAX, TOL_MIN, TOL_MAX)

        return np.array([penalty, tr_rad_path, tol_path]).reshape([-1, 1])

    def boundary_hyperparam_init(self):
        penalty = PEN_INIT
        tr_rad = ALPHA
        tol_leg = np.clip((1 / (2 * penalty)) ** BETA * TOL_MAX, TOL_MIN, TOL_MAX)

        return np.array([penalty, tr_rad, tol_leg]).reshape([-1, 1])

    def test_adjust(self, Jac_stack, prev_b_moments, hyper_param, epi_num, num_reject):
        """
        :param Jac_stack: "path" --> np.c_[np.r_[Hu, Hxu], np.r_[Hl, Hxl]]
                         "term" -->  np.r_[Pm, Pxm]
        :param prev_b_moments:
        :param penalty:
        :param Jac_tol:
        :param epi_num:
        :return:
        """

        penalty, tr_rad, Jac_tol = hyper_param
        # 1. 1st order convergence: Hl, Hu = 0
        Jac = Jac_stack[:, 0]
        Hu, Hl = Jac[:self.a_dim], Jac[self.a_dim:]
        Hl_norm = np.linalg.norm(Hl) / len(Hl) # Feasibility
        Hu_norm = np.linalg.norm(Hu) / len(Hu) # Optimality
        Jac_norm = np.linalg.norm(Jac) / len(Jac) # Feasibility + Optimality

        # 2. Convergence test & Adjust convergence tolerance
        if Jac_norm < Jac_tol:
            converged = True
            Jac_tol = np.clip((1 / (2 * penalty)) ** BETA * Jac_tol, TOL_MIN, TOL_MAX)
        else:
            converged = False
            Jac_tol = np.clip((1 / (2 * penalty)) ** BETA * TOL_MAX, TOL_MIN, TOL_MAX)

        # 3. Adjust penalty parameter
        if not converged:  # Fail: Increase penalty
            penalty = np.clip((1 + PEN_INC_RATE) * penalty, PEN_MIN, PEN_MAX)
        else:
            penalty = np.clip((1 - PEN_DEAC_RATE) * penalty, PEN_MIN, PEN_MAX)

        # 4. Update ADAM moments: Separate 1st & 2nd moments
        m_b, v_b, m_u, v_u = self.adam_update(prev_b_moments, Jac_stack, epi_num)
        # ub_1st_mom = m_u
        ub_1st_mom = Jac_stack
        new_b_moments = [m_b, v_b]  # Updated biased 1st and 2nd moments

        # 5. Trust region step size update
        min_sche = TR_MIN * (1 - 0.9 * epi_num / self.epi_max)
        max_sche = TR_MAX * (1 - 0.9 * epi_num / self.epi_max)

        # Learning rate adjust (RMSProp)
        alpha_decay = ALPHA / np.sqrt(epi_num + 1) * 2 ** (-num_reject)
        TR_radius = np.clip(alpha_decay * np.linalg.norm(m_u) / (np.sqrt(v_u) + EPS), min_sche, max_sche)  # Learning rate adjust (RMSProp)

        opt_param = np.array([penalty.item(), TR_radius.item(), Jac_tol.item()]).reshape([-1, 1])
        return opt_param, [Hu_norm, Hl_norm], ub_1st_mom, new_b_moments

    def adam_update(self, prev_b_moments, Jac_stack, epi_num):
        # 1st moments --> Forward step size adjustment
        # 2nd moments --> TR radius adjustment
        m_b, v_b = prev_b_moments  # Biased 1st and 2nd moments

        # Weighted update of biased moments
        m_b = BETA1 * m_b + (1 - BETA1) * Jac_stack
        v_b = BETA2 * v_b + (1 - BETA2) * np.linalg.norm(Jac_stack) ** 2

        # Bias correction
        m_u = m_b / (1 - BETA1 ** (epi_num + 1))
        v_u = v_b / (1 - BETA2 ** (epi_num + 1))

        return m_b, v_b, m_u, v_u