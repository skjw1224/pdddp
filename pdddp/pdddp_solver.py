import numpy as np


from pdddp.trust_region_optimizer import TrustRegionOpt
from pdddp.constraint_augment import ConstraintAug
from pdddp.hyperparameter import HyperParameter


""" v2
    - Add Trust region optimization
    - Time normalization of value functions
    - Terminal cost differentiated to the stage-wise cost
    v3: constraint optimal control
    - Augmented Lagrangian (Powell-Hestenes-Rockafeller) method
    - Backward HJB for PHR method
    - Inner iteration and convergence test
    - Batch structure modification for inner iteration hyperparameter
"""

# SAFETY_GUARDS
X_MAX = 1
DX_MAX = 0.1
U_MAX = 1
LAGR_MAX = 20
DYNS_MAX = 10
PCOST_MAX = 10
PCONST_MAX = 10
COST_MAX = 10
CONST_MAX = 10
V_MAX = 20

class CDDPSolver(object):
    def __init__(self, env, epi_max):
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.c_dim_list = env.c_dim
        self.cPineq_dim, self.cPeq_dim, self.cTineq_dim, self.cTeq_dim = self.c_dim_list
        self.cP_dim = self.cPineq_dim + self.cPeq_dim
        self.cT_dim = self.cTineq_dim + self.cTeq_dim

        # Nonnegative Lagrangian for inequality constraints, equality constraints --> LB: -LAGRMAX
        self.LAGR_MIN_Pineq_masked = np.concatenate([np.zeros([self.cPineq_dim, 1]), -LAGR_MAX * np.ones([self.cPeq_dim, 1])])
        self.LAGR_MIN_Tineq_masked = np.concatenate([np.zeros([self.cTineq_dim, 1]), -LAGR_MAX * np.ones([self.cTeq_dim, 1])])

        self.dt = env.dt
        self.time_norm = env.tT - env.t0 # Time denormalizing factor  [0, 1] --> [t0, tT]
        self.time_norm = 1.
        self.zero_center_scale = env.zero_center_scale

        self.X_MAX, self.U_MAX = X_MAX, U_MAX
        if self.zero_center_scale == True:
            self.X_MIN, self.U_MIN = -X_MAX, -U_MAX
        else:
            self.X_MIN, self.U_MIN  = 0, 0
        self.DX_MAX = DX_MAX / env.nT
        self.V_MAX = V_MAX * env.nT

        self.path_sym_args, self.term_sym_args = env.path_sym_args, env.term_sym_args
        self.dyns_fncs, self.cost_fncs, self.costT_fncs, self.constP_fncs, self.constT_fncs = env.model_derivs
        self.I_fnc = env.I_fnc

        # self.critic = critic
        # self.lagrangian = lagrangian

        self.x_idx_upper = np.triu_indices(self.s_dim)
        self.l_idx_upper = np.triu_indices(self.cP_dim)

        self.compute_MS_alg = True
        self.trust_region_opt = TrustRegionOpt(self.s_dim, self.a_dim, self.c_dim_list)
        self.constraint_aug = ConstraintAug(self.s_dim, self.a_dim, self.c_dim_list)
        self.hyper_parameter = HyperParameter(self.s_dim, self.a_dim, self.c_dim_list, epi_max)

        self.epi_num = 0
        self.num_reject = 0

    def assign_epi_num(self, epi_num, num_reject):
        self.epi_num = epi_num
        self.num_reject = num_reject


    def bwd_path_eval(self, env_data, new_Vnplus_list, prev_b_moments):
        """Backward single step computation: 1. Solve primal QCQP under dx, l = 0 --> Compute du, l
                                             2. Evaluate dyn, cost, const function at primal perturbed (u + du, l) point
                                             3. Solve dual QCQP under u+du, l , dx=0 --> Compute dl
                                             4. Evaluate FB/FF gains under u+du, l+dl, dx=0
                                             5. Backward HJB equation: V
                                             6. Backward adjoint equations for dynamics sensitivity: F_leg """

        # Extract arguments for env function
        xcurr, xest, u, Lagr, hyper_param_path, p_mu, p_sigma, p_eps = env_data
        path_eval_args = [xest, u, p_mu, p_sigma, p_eps]

        Vplus_fncs = [np.clip(Vi, -self.V_MAX, self.V_MAX) for Vi in new_Vnplus_list]

        dyns_eval = [np.clip(fi, -DYNS_MAX, DYNS_MAX) for fi in self.dyns_fncs(*path_eval_args)]  # F, Fx, Fu
        cost_eval = [np.clip(fi, -PCOST_MAX, PCOST_MAX) for fi in self.cost_fncs(*path_eval_args)]  # L, Lx, Lu, Lxx, Lxu, Luu
        constP_eval = [np.clip(fi, -PCONST_MAX, PCONST_MAX) for fi in self.constP_fncs(*path_eval_args)]  # G, Gx, Gu, Gxx, Gxu, Guu

        penalty_path, tr_radius_path, tol_path = hyper_param_path

        # 1. Solve primal QCQP (Corrector): Solve (delu, lambda) under (delx, lambda) = (0, 0)
        # Only penalty augment (no Lagr augment)
        # Use zero value (MX.zeros) for Lagrangian argument for primal estimation of Lagrangian

        aug_primal_list, active_set = self.constraint_aug.aug_constfncs(constP_eval, Lagrangian=np.zeros([self.cP_dim, 1]),
                                                                        penalty=penalty_path, const_type='path')

        Hamiltonian_list = self.eval_Hamiltonian(Vplus_fncs, dyns_eval, cost_eval, aug_primal_list, eval_all=False)
        Hu, _, Huu, _, _ = Hamiltonian_list
        delu, lagr, u_Hess_corr = self.trust_region_opt.solve_primal_QCQP(Huu, Hu, constP_eval, np.zeros([self.cP_dim, 1]), active_set,
                                                                          hyper_param_path, const_type='path')

        lagr = np.clip(lagr, self.LAGR_MIN_Pineq_masked, LAGR_MAX)

        # 2. Solve dual QCQP (Dual update): Recompute Hamiltonian and active set under nominal condition (u, lambda), delx = 0
        aug_p_pb_list, active_set = self.constraint_aug.aug_constfncs(constP_eval, lagr, penalty_path, const_type='path')

        Hamiltonian_list = self.eval_Hamiltonian(Vplus_fncs, dyns_eval, cost_eval, aug_p_pb_list, eval_all=False)
        Hu, Hl, Huu, Hul, Hll = Hamiltonian_list

        # Dual Hessian & Jacobian correction
        Huu_tilde = Huu + u_Hess_corr
        Huu_ul = self.trust_region_opt.gain_eval(Huu_tilde, Hul, None, prob_type='min')
        Hll_p = Hll + Huu_ul.T @ Hul
        Hl_p = Hl + Huu_ul.T @ Hu

        dell, l_Hess_corr, _ = self.trust_region_opt.solve_dual_QCQP(Hll_p, Hl_p, lagr,
                                                                     self.cPineq_dim, hyper_param_path, const_type='path')

        # 3. Predictor: Eval FF / FB gains under (x, u + delu, lambda + dell) (Under perturbed condition)
        x, u, pm, ps, pe = path_eval_args
        delu = delu + Huu_ul @ dell
        new_u = u + delu
        new_lagr = lagr + dell

        aug_pd_pb_list, active_set_list = self.constraint_aug.aug_constfncs(constP_eval, new_lagr, penalty_path, const_type='path')

        Hamiltonian_list = self.eval_Hamiltonian(Vplus_fncs, dyns_eval, cost_eval, aug_pd_pb_list)
        H, Hx, Hu, Hl, Hxx, Hxu, Hxl, Huu, Hul, Hll, La = Hamiltonian_list

        # Hessian TR correction & Jacobian nonneg ineq constr Lagr correction
        Huu_tilde = Huu + u_Hess_corr
        Hll_tilde = Hll + l_Hess_corr

        # Eval FF / FB gains
        ul_Jac_stack = np.r_[np.c_[Hu, Hxu.T], np.c_[Hl, Hxl.T]]
        ul_Mat = np.r_[np.c_[Huu_tilde, Hul], np.c_[Hul.T, Hll_tilde]]

        path_gain_stack = self.trust_region_opt.gain_eval(ul_Mat, ul_Jac_stack, self.a_dim, prob_type='minmax')

        opt_param, path_Jac_norm, path_ubJac_mom, H_b_moments = \
            self.hyper_parameter.test_adjust(path_gain_stack, prev_b_moments, hyper_param_path, self.epi_num, self.num_reject)

        "Solve backward HJB ODE & Optimize u"
        path_gain_list = np.split(path_ubJac_mom, np.cumsum([1]), axis=1) # l, Kx
        Vminus_list = self.backward_HJB(Vplus_fncs, Hamiltonian_list, path_gain_list)

        Vminus_list = [np.clip(Vi, -self.V_MAX, self.V_MAX) for Vi in Vminus_list]

        # Update Lagrangian & Hyperparameters
        updated_env_data = [xcurr, xest, u, new_lagr, opt_param, p_mu, p_sigma, p_eps]

        primal_pb_args = [x, new_u, pm, ps, pe]  # Perturbed argument

        dyns_pb_list = [np.clip(fi, -DYNS_MAX, DYNS_MAX) for fi in self.dyns_fncs(*primal_pb_args)]
        const_pb_list = [np.clip(fi, -PCONST_MAX, PCONST_MAX) for fi in self.constP_fncs(*primal_pb_args)]
        cost_pb_list = [np.clip(fi, -PCOST_MAX, PCOST_MAX) for fi in self.cost_fncs(*primal_pb_args)]

        aug_pd_pb_list, active_set_list = self.constraint_aug.aug_constfncs(const_pb_list, new_lagr, penalty_path,
                                                                            const_type='path')

        Hamiltonian_list = self.eval_Hamiltonian(Vplus_fncs, dyns_pb_list, cost_pb_list, aug_pd_pb_list, eval_all=False)
        Hu, Hl, _, _, _ = Hamiltonian_list
        path_Jac_norm = [np.linalg.norm(Hu) / len(Hu), np.linalg.norm(Hl) / len(Hl)]

        return Vminus_list, H, path_gain_list, opt_param, path_Jac_norm, H_b_moments, updated_env_data

    def fwd_single_step(self, env_data, path_gain_list, delx_prev):
        """Compute forward delx sensitivity given FF/FB ctrl/const gains & Update x, l, u for single time step
        args:
        feed_gain_list: FF/FB ctrl/const gains at time t
        feed_gain_list_master: FF/FB collocation gains at leg terminal time
        delx_prev: delx at previous time step"""
        # delx_prev: None --> Step based method, update xest
        # delx_prev: Have value --> Episode based method, don't update xest

        xcurr, xest, u, Lagr, hyper_param_path, p_mu, p_sigma, p_eps = env_data

        if delx_prev is None:
            delx_prev = np.zeros([self.s_dim, 1])
        delx_prev = np.clip(delx_prev, -self.DX_MAX, self.DX_MAX)

        "solve forward del_system"
        delx_plus, del_u, del_Lagr = self.forward_dyns_eval(delx_prev, path_gain_list, env_data)

        new_xplus = xest + delx_plus
        new_u = u + del_u
        new_Lagr = Lagr
        new_Lagrn, del_Lagrn = new_Lagr, del_Lagr


        return new_xplus, delx_plus, new_u, new_Lagrn

    def backward_HJB(self, Vplus_list, H_list, path_gain_list):
        """Backward HJB ODE for single time step: Compute Vnminus for optimized u"""
        #if self.compute_MS_alg is True: s - related vars: dummy zero values
        V_plus, Vx_plus, Vxx_plus = Vplus_list
        H, Hx, Hu, Hl, Hxx, Hxu, Hxl, Huu, Hul, Hll, La = H_list
        if path_gain_list: # gain for u and l
            l, Kx = path_gain_list # (lu, ll); (Kux, Klx);
            Hv = np.r_[Hu, Hl]
            Hxv = np.c_[Hxu, Hxl]
            Hmat = np.r_[np.c_[Huu, Hul], np.c_[Hul.T, Hll]]

        H_reduc, Hx_reduc, Hxx_reduc = 0., 0., 0.
        if path_gain_list: # update u
            H_reduc = l.T @ Hv + 0.5 * l.T @ Hmat @ l
            Hx_reduc = Hxv @ l + Kx.T @ Hmat @ l + Kx.T @ Hv
            Hxx_reduc = Hxv @ Kx + Kx.T @ Hmat @ Kx + Kx.T @ Hxv.T

        # Continuous-time
        V_minus = V_plus + (La + H_reduc) * self.dt         # 1*1
        Vx_minus = Vx_plus + (Hx + Hx_reduc) * self.dt      # 1*S
        Vxx_minus = Vxx_plus + (Hxx + Hxx_reduc) * self.dt  # S*S

        # Symmetrize
        Vxx_minus = (Vxx_minus + Vxx_minus.T) / 2

        Vminus_list = [V_minus, Vx_minus, Vxx_minus]

        return Vminus_list

    def backward_HJB_boundary(self, Vbc_list, bc_feed_gain):
        """Backward HJB equation for boundary conditions (Leg, terminal: Compute Pminus for optimized mu (terminal ocnstraint or leg constraint)
        Args:
            Vbc_list: P_(xm)
            bc_feed_gain: Km_(x) (Term)
        Output:
            Pminus list: P_(x) (Term)
        """

        P, Px, Pm, Pxx, Pxm, Pmm = Vbc_list
        lm, Kmx = bc_feed_gain

        P_reduc = Pm.T @ lm + 0.5 * lm.T @ Pmm @ lm
        Px_reduc = Pxm @ lm + Kmx.T @ Pmm @ lm + Kmx.T @ Pm
        Pxx_reduc = Pxm @ Kmx + Kmx.T @ Pmm @ Kmx + Kmx.T @ Pxm.T

        P_minus = P + P_reduc
        Px_minus = Px + Px_reduc
        Pxx_minus = Pxx + Pxx_reduc

        Pxx_minus = (Pxx_minus + Pxx_minus.T) / 2

        Pminus_list = [P_minus, Px_minus, Pxx_minus]

        return Pminus_list

    def forward_dyns_eval(self, delx_prev, path_gain_list, env_data):
        """Discrete-time ODE for a single time step: Compute delu given FF/FB gains w.r.t. s, mu, lambda, u --> Compute delx_plus"""

        x, xest, u, Lagr, hyper_param_path, pm, ps, pe = env_data

        delx = delx_prev            # (1*S)
        l, Kx = path_gain_list  # [lu, ll]; [Kux, Klx];

        delul = l + Kx @ delx

        delu, dell = delul[:self.a_dim, :], delul[self.a_dim:, :]
        delu = np.clip(u + delu, self.U_MIN, self.U_MAX) - u

        dell = np.clip(Lagr + dell, self.LAGR_MIN_Pineq_masked, LAGR_MAX) - Lagr # Inequality lagrangian: non-negative

        new_x = x + delx
        new_u = u + delu

        # 1st order Euler ODE for new point
        xplus = np.clip(self.I_fnc(x0=x, p=u)['xf'].full(), self.X_MIN, self.X_MAX)
        delx_plus = np.clip(self.I_fnc(x0=new_x, p=new_u)['xf'].full(), self.X_MIN, self.X_MAX) - xplus

        return delx_plus, delu, dell

    def eval_Hamiltonian(self, value_fncs, dyns_fncs, cost_fncs, cost_aug_fncs, eval_all=True):
        """Evaluate Hamiltonian in single time step: H, Hjac, Hhess"""
        F, Fx, Fu = [dyns_fncs[0], dyns_fncs[1], dyns_fncs[2]]  # (S*1), (S*S), (S*A), S*(S*S), S*(S*A), S*
        L, Lx, Lu, Lxx, Lxu, Luu = cost_fncs  # (1,1), (S*1), (A*1), (S*S), (S*A), (A*A)
        P, Px, Pu, Pl, Pxx, Pxu, Pxl, Puu, Pul, Pll = cost_aug_fncs

        # Cost augmentation
        La = L + P  # (1*1)
        Lax = Lx + Px  # (S*1)
        Lau = Lu + Pu  # (A*1)
        Lal = Pl  # (C*1)
        Laxx = Lxx + Pxx  # (S*S)
        Laxu = Lxu + Pxu  # (S*A)
        Laxl = Pxl  # (S*C)
        Lauu = Luu + Puu  # (A*A)
        Laul = Pul  # (C*A)
        Lall = Pll  # (C*C)

        V, Vx, Vxx = value_fncs  # (1,1), (S*1), (S*S)

        # Continuous-time Hamiltonians
        H = La + F.T @ Vx  # (1*1)
        Hx = Lax + Fx.T @ Vx  # (S*1)
        Hu = Lau + Fu.T @ Vx  # (A*1)
        Hl = Lal  # (C*1)
        Hxx = Laxx + Vxx @ Fx + Fx.T @ Vxx #+ sum([Vxi @ Fxxi for Vxi, Fxxi in zip(Vx, Fxx)]) # (S*S)
        Hxu = Laxu + Vxx.T @ Fu #+ sum([Vxi @ Fxui for Vxi, Fxui in zip(Vx, Fxu)]) # (S*A)
        Hxl = Laxl  # (S*C)
        Huu = Lauu #+ sum([Vxi @ Fuui for Vxi, Fuui in zip(Vx, Fuu)]) # (A*A)
        Hul = Laul  # (A*C)
        Hll = Lall  # (C*C)


        if eval_all is False:
            return [Hu, Hl, Huu, Hul, Hll]
        else:
            return [H, Hx, Hu, Hl, Hxx, Hxu, Hxl, Huu, Hul, Hll, La]

    def bwd_term_eval(self, env_data, prev_b_moments):
        x, xest, Lagr, hyper_param_term, p_mu, p_sigma, p_eps = env_data
        # Extract arguments for env functions
        term_eval_args = [x, p_mu, p_sigma, p_eps]
        penalty_term, tr_radius_term, tol_term = hyper_param_term

        self.compute_MS_alg = False

        costT_eval = [np.clip(fi, -COST_MAX, COST_MAX) for fi in self.costT_fncs(*term_eval_args)]  # LT, LTx, LTxx
        constT_eval = [np.clip(fi, -CONST_MAX, CONST_MAX) for fi in self.constT_fncs(*term_eval_args)]  # GT, GTx, GTxx
        LT, LTx, LTxx = costT_eval

        # 1. Solve primal QP (Corrector): Estimate Lagrangian (m) under (delxT, m) = (0, 0)
        # Only penalty augment (no Lagr augment), no need to distinguish strict, weak feas
        # casadi function for constraint augmentation and active set evaluation
        _, active_set = self.constraint_aug.aug_constfncs(constT_eval, Lagrangian=np.zeros([self.cT_dim, 1]),
                                                          penalty=penalty_term, const_type='terminal')
        lagr_term = self.trust_region_opt.solve_primal_terminal(constT_eval[0], active_set, hyper_param_term)

        # 2. Solve dual QCQP (Dual update): Recompute constr augmentation, active set under nominal condition (m), delxT = 0
        aug_primal_list, _ = self.constraint_aug.aug_constfncs(constT_eval, lagr_term, penalty_term, const_type='terminal')
        _, _, _, Pm, _, _, _, _, _, Pmm = aug_primal_list

        delm, m_Hess_corr, m_Jac_corr = self.trust_region_opt.solve_dual_QCQP(Pmm, Pm, lagr_term, self.cTineq_dim, hyper_param_term, const_type='terminal')
        lagr_term = np.clip(lagr_term, self.LAGR_MIN_Tineq_masked, LAGR_MAX)

        # 3. Predictor: Eval FF/FB gains under (x, m + delm): Do not recompute constr aug & active set under (m + delm) --> Already corrected above
        new_lagr_term = lagr_term + delm

        # Dual Hess & Jacobian correction
        aug_d_pb_list, _ = self.constraint_aug.aug_constfncs(constT_eval, new_lagr_term, penalty_term, const_type='terminal')
        P, Px, _, Pm, Pxx, _, Pxm, _, _, Pmm = aug_d_pb_list
        P = P + LT
        Px = Px + LTx
        Pxx = Pxx + LTxx
        Pmm_tilde = Pmm + m_Hess_corr
        Pm_tilde = Pm + m_Jac_corr

        m_Jac_stack = np.c_[Pm_tilde, Pxm.T]

        term_gain_stack = self.trust_region_opt.gain_eval(Pmm_tilde, m_Jac_stack, None, prob_type='max')

        # Update hyperparameter, biased moment & Evaluate unbiased Jacobian moment
        opt_param, term_Jac_norm, term_ubJac_mom, b_moments = \
            self.hyper_parameter.test_adjust(term_gain_stack, prev_b_moments, hyper_param_term, self.epi_num, self.num_reject)

        term_gain_list = np.split(term_ubJac_mom, [1], axis=1) #lm, Kmx

        # Backward equation for boundary cond'n for terminal time: eliminate m
        VT, VxT, VxxT = self.backward_HJB_boundary([P, Px, Pm, Pxx, Pxm, Pmm], term_gain_list)

        # Normalization
        Vterm_list = [VT, VxT, VxxT]
        Vterm_list = [np.clip(Vi, -self.V_MAX, self.V_MAX) for Vi in Vterm_list]

        # Update Lagrangian & Hyperparameters
        updated_env_data = [x, xest, new_lagr_term, opt_param, p_mu, p_sigma, p_eps]

        return Vterm_list, term_gain_list, opt_param, term_Jac_norm, b_moments, updated_env_data

    def eval_boundary_conditions_at_term_fwd(self, env_data, delxT, boundary_gain_list):
        # m: Terminal constraint Lagrangian (if final leg is True)

        x, xest, Lagr, hyper_param_term, p_mu, p_sigma, p_eps = env_data

        lm, Kmx = boundary_gain_list
        delm = lm + Kmx @ delxT
        delm = np.clip(Lagr + delm, self.LAGR_MIN_Tineq_masked, LAGR_MAX) - Lagr

        new_m = Lagr + delm

        return new_m, delm