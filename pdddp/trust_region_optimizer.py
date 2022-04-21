import numpy as np
import scipy as sp
import scipy.linalg
import casadi as ca

Du = 1
Dl = 0.1

class TrustRegionOpt(object):
    def __init__(self, s_dim, a_dim, c_dim_list):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.cPineq_dim, self.cPeq_dim, self.cTineq_dim, self.cTeq_dim = c_dim_list
        self.cP_dim = self.cPineq_dim + self.cPeq_dim
        self.cT_dim = self.cTineq_dim + self.cTeq_dim

        self.M = 100000 # Arbitrary large

        self.primal_QCQP_path = self.def_primal_QCQP(self.a_dim, self.cP_dim)
        self.dual_QCQP_path = self.def_dual_QCQP(self.cP_dim)
        self.dual_QCQP_term = self.def_dual_QCQP(self.cT_dim)

    def def_primal_QCQP(self, nu, nc):
        delu_sym = ca.MX.sym('delu_qcqp', nu)
        slack_sym = ca.MX.sym('slack_qcqp', nc)
        Hess_sym = ca.MX.sym('Hess_qcqp', nu, nu)
        Jac_sym = ca.MX.sym('Jac_qcqp', nu)
        A_mask_sym = ca.MX.sym('A_Mask_qcqp', nc)
        G_sym = ca.MX.sym('G_qcqp', nc)
        Gu_sym = ca.MX.sym('Gc_qcqp', nc, nu)
        hypp_sym = ca.MX.sym('hypp_qcqp', 3)
        penalty, radius, tol = ca.vertsplit(hypp_sym)

        D = Du * ca.MX.eye(nu)
        c = penalty * ca.MX.eye(nc)

        qcqp = {}
        qcqp['x'] = ca.vertcat(delu_sym, slack_sym)
        qcqp['p'] = ca.vertcat(ca.reshape(Hess_sym, nu ** 2, 1), Jac_sym, A_mask_sym, G_sym,
                               ca.reshape(Gu_sym, nc * nu, 1), hypp_sym)
        qcqp['f'] = 0.5 * delu_sym.T @ Hess_sym @ delu_sym + delu_sym.T @ Jac_sym \
                    + 0.5 * ((1 - A_mask_sym) * slack_sym).T @ c @ ((1 - A_mask_sym) * slack_sym)
        # 1. Trust region constraint |D @ dell_sym| <= r
        # 2. Feasibility under primal perturbation
        qcqp['g'] = ca.vertcat(delu_sym.T @ D.T @ D @ delu_sym - radius ** 2,
                               G_sym + Gu_sym @ delu_sym - (1 - 2 * A_mask_sym) * slack_sym)

        opts = {'print_time': False, "ipopt": {"max_iter": 300, 'print_level': 0}}  # print_level: 0 (Nothing) 5 (default) 12 (max)
        S = ca.nlpsol('S', 'ipopt', qcqp, opts)
        return S

    def def_dual_QCQP(self, nc):
        D = Dl * ca.MX.eye(nc)

        # Nlpsol variable, parameter should be column vector
        dell_sym = ca.MX.sym('dell_sym', nc)
        Hess_sym = ca.MX.sym('Hess', nc, nc)
        Jac_sym = ca.MX.sym('Jac', nc)
        Lagr_sym = ca.MX.sym('Lagr_primal', nc)
        hypp_sym = ca.MX.sym('hypp_qcqp', 3)
        penalty, radius, tol = ca.vertsplit(hypp_sym)

        qcqp = {}
        qcqp['x'] = dell_sym
        qcqp['p'] = ca.vertcat(ca.reshape(Hess_sym, nc ** 2, 1), Jac_sym, Lagr_sym, hypp_sym)
        qcqp['f'] = 0.5 * dell_sym.T @ Hess_sym @ dell_sym + dell_sym.T @ Jac_sym
        # 1. Trust region constraint |D @ dell_sym| <= r
        # 2. Nonnegative lagrangian for inequality constraints
        qcqp['g'] = ca.vertcat(dell_sym.T @ D.T @ D @ dell_sym - radius ** 2,
                               -dell_sym - Lagr_sym)

        opts = {'print_time': False, "ipopt": {"max_iter": 300, 'print_level': 0}}  # print_level: 0 (Nothing) 5 (default) 12 (max)
        S = ca.nlpsol('S', 'ipopt', qcqp, opts)
        return S

    def solve_primal_QCQP(self, Hess, Jac, constr, Lagr, active_set, hyper, const_type):
        # path constraint: cP_dim, a_dim
        if const_type == 'path':  # cP, u
            nc, nu = self.cP_dim, self.a_dim
            QCQP = self.primal_QCQP_path

        c = hyper[0]
        G, Gu = constr[0] + Lagr / c, constr[2]
        D = Du * np.eye(nu)

        # Feasible --> 1
        # Infeasible (Infeas ineq, equality) --> 0
        A_Mask = np.sign(active_set)

        # LB: unbounded for feasible constraint, 0 for infeasible constraint
        lbg = np.r_[-self.M * np.ones([1, 1]), np.zeros([nc, 1])]
        # UB: 0 for all constraints (eps-relaxed)
        ubg = np.zeros([1 + nc, 1])

        # Optimize with numeric values
        primal_qcqp = QCQP(p=np.concatenate([np.reshape(Hess, [nu**2, 1]), Jac, A_Mask, G, np.reshape(Gu, [nc*nu, 1]), hyper]), lbg=lbg, ubg=ubg)

        delu_eval = primal_qcqp['x'][:nu].full()
        tr_dual, lagr = np.vsplit(primal_qcqp['lam_g'], [1])

        Hess_corr_eval = 2 * tr_dual * D.T @ D
        lagr_eval = np.abs(lagr)

        return delu_eval, lagr_eval, Hess_corr_eval

    def solve_dual_QCQP(self, Hess, Jac, Lagr, nineq, hyper, const_type):
        # path constraint: cP_dim, a_dim
        if const_type == 'path':  # cP
            nc = self.cP_dim
            QCQP = self.dual_QCQP_path
        elif const_type == 'terminal':  # cT
            nc = self.cT_dim
            QCQP = self.dual_QCQP_term

        D = Dl * np.eye(nc)
        Hess = -Hess
        Jac = -Jac

        Eq_Mask = np.concatenate([np.zeros([nineq, 1]), np.ones([nc - nineq, 1])])
        # UB: 0 UBs for inequality constraints (nineq), others (nc - nineq) are unbounded (relaxed by M)
        ubg = np.concatenate([np.zeros([1, 1]), self.M * Eq_Mask])

        # Optimize with numeric values
        dual_qcqp = QCQP(p=np.concatenate([np.reshape(Hess, [nc**2, 1]), Jac, Lagr, hyper]), ubg=ubg)

        dell_eval = dual_qcqp['x'].full()
        tr_dual, nonneg_dual = np.vsplit(dual_qcqp['lam_g'], [1])

        Hess_corr = 2 * tr_dual * D.T @ D
        Jac_corr = nonneg_dual

        # Negative correction
        Hess_corr_eval = -Hess_corr
        Jac_corr_eval = -Jac_corr

        return dell_eval, Hess_corr_eval, Jac_corr_eval

    def solve_primal_terminal(self, constr, active_set, hyper):
        c_dim = constr.shape[0]
        penalty, radius, tol = hyper.flatten()

        # Feasible (1)
        # Infeasible (Infeas ineq, equaliy) (0)
        lagr = np.zeros([c_dim, 1])
        for i in range(c_dim):
            if active_set[i] == 0:
                lagr[i] = penalty * constr[i]

        return lagr

    def gain_eval(self, Mat, Vec_stack, minmax_slice_idx, prob_type):
        """args:
        Mat: Hessian: ex) Huu or [Huu, Hul; Hlu, Hll]
        Vec_stack: Row-stack Jacobian list ex) Hu, Hxu, Hsu, Hlu
        gain_slice_idx: Vec_stack column slicing index: split into lu, Kux, Hus, Hun
        minmax_slice_idx: Optional row slicing index for minmax problem: split into vec_stack_min & vec_stack_max
        prob_type: min, max, minmax
        out: gain_list --> List of FF/FB gain matrix ex) l, Kx, Ks, Kl """

        if prob_type == 'minmax':
            gain_stack = self.gain_eval_minmax(Mat, Vec_stack, minmax_slice_idx)
            return gain_stack

        elif prob_type == 'max' :
            gain_stack = self.solve_uc_QP_np(-Mat, -Vec_stack)
        elif prob_type == 'min':
            gain_stack = self.solve_uc_QP_np(Mat, Vec_stack)

        return gain_stack

    def gain_eval_minmax(self, Mat, Vec_stack, minmax_slice_idx):

        Qmm, QmM, QMm, QMM = Mat[:minmax_slice_idx, :minmax_slice_idx], Mat[:minmax_slice_idx, minmax_slice_idx:], \
                             Mat[minmax_slice_idx:, :minmax_slice_idx], Mat[minmax_slice_idx:, minmax_slice_idx:]
        Vec_stack_m, Vec_stack_M = Vec_stack[:minmax_slice_idx, :], Vec_stack[minmax_slice_idx:, :] # min var vector stack + max var vector stack, Vec_stack_M: negative


        m2M_gain = self.solve_uc_QP_np(Qmm, QmM)
        M2m_gain = self.solve_uc_QP_np(-QMM, -QMm) # Max problem

        # Shifted hessians due to min max prob
        Qmm_minmax = Qmm + m2M_gain @ QMm
        QMM_minmax = QMM + M2m_gain @ QmM

        # Shifted jacobians due to min max prob
        Vec_stack_m_minmax = Vec_stack_m + m2M_gain @ Vec_stack_M
        Vec_stack_M_minmax = Vec_stack_M + M2m_gain @ Vec_stack_m

        # FF/FB gain stacked
        feed_gain_m = self.solve_uc_QP_np(Qmm_minmax, Vec_stack_m_minmax)
        feed_gain_M = self.solve_uc_QP_np(-QMM_minmax, -Vec_stack_M_minmax) # Max problem

        # FF/FB gain splited to list  [lm, lM]; [Kmx, KMx]; [Kms, KMs]; [Kmn, KMn]
        feed_gain_stack = np.r_[feed_gain_m, feed_gain_M]

        return feed_gain_stack

    def solve_uc_QP_np(self, Hess, Jac_stack):
        """args;
        Hess: Positive SD matrix
        Jac: Feedforward jacobian for QCQP optimization (ex: Hu) - row vector
        RHS_vec_stack (ex: [Hu; Hxu; Hsu; Hnu] or QMm) - row stack
        output:
        gain_stack: Row-stacked  FF/FB gain matrix (Shape conserved)"""

        if np.linalg.norm(Hess) > 1E-6:
            try:
                U = sp.linalg.cholesky(Hess)  # -Huu_inv @ [Hu, Hux]
                gain_stack = - sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, Jac_stack, lower=True))  # solve_triangular --> arg: column vector -> Jac_stack.T --> # Transpose to Row vector again
            except np.linalg.LinAlgError:
                gain_stack = - np.linalg.inv(Hess) @ Jac_stack
        else:  # Hessian = zero matrix --> Linear update
            gain_stack = np.zeros_like(Jac_stack)  # Gain = zero vec
        return gain_stack
