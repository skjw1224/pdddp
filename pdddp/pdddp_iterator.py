import numpy as np
from pdddp.pdddp_solver import CDDPSolver

MAX_ITER = 1
LAGR_ITER_PER_U = 1

class CDDPIterator(CDDPSolver):
    def __init__(self, env, epi_max, leg_idx):
        super().__init__(env, epi_max)
        self.leg_idx = leg_idx
        self.init_TRmoments = [[[np.zeros([env.a_dim + env.cPineq_dim + env.cPeq_dim, 1 + env.s_dim]), 0.] for _ in range(leg_idx[-1])],
                               [[0., 0.]]]

        self.feas_filter = np.inf


    def solve_cddp(self, epi_traj, prev_b_moments, epi_num, save_vfncs):
        """args: epi_path_data, epi_term_data = epi_data
           epi path data: x, x2, u, l, [c, tr, tol], pm, ps, pe
           epi term data: xT, mT, [c, tr, tol], pm, ps, pe
           prev_TRmoments: 1st and 2nd moments (ADAM opt) of previous episode H, VT -- List of [1st, 2nd]

           Output: epi_path_solutions, epi_term_solutions = epi_value_gains
                   new TR moments: H, VT
            epi path solution: Vn reshape(Mat2Vec) list, FF/FB gain_list, new u, new l, new [c, tr, tol], Augmented Cost, Convergence Jac norm (Hu, Hl)
            epi term solution: VnT reshape(Mat2Vec) list, FF/FB gain_list, new mT, new  [c, tr, tol], Convergence Jac norm (PmT)"""

        self.assign_epi_num(epi_num)
        epi_path_data, epi_term_data = epi_traj
        if prev_b_moments is None:
            prev_b_moments = self.init_TRmoments
        prev_path_bmom_epi, prev_term_bmom_epi = prev_b_moments

        "Backward sweep"
        Vn_list_epi, VnT_list_epi = [], []
        H_epi = []
        path_gain_list_epi, term_gain_list_epi = [], []
        path_bmom_epi, term_bmom_epi = [], []
        new_hypparam_epi, new_hypparam_term_epi = [], []
        conv_stat_path_epi, conv_stat_term_epi = [], []

        # Initialize outer loop by assigning terminal condition
        # Leg and terminal BC conditions
        Vnplus_list, term_gain_list, hyp_param_term, VnTmT_norm, VnTbmom, epi_term_data[0] = \
            self.bwd_term_eval(env_data=epi_term_data[0], prev_b_moments=prev_term_bmom_epi[0])
        # Save terminal data: VT, terminal gain, optpar_term, VnT_moment, conv_stat_term
        if save_vfncs:
            VnT_list_epi.append(Vnplus_list)
        term_gain_list_epi.append(term_gain_list)
        new_hypparam_term_epi.append(hyp_param_term)
        term_bmom_epi.append(VnTbmom)
        conv_stat_term_epi.append(VnTmT_norm)

        # Path conditions
        for i in reversed(range(self.leg_idx[0], self.leg_idx[1])):
            Vnminus_list, Hamilt, path_gain_list, hyp_param, Hul_norm, Hbmom, epi_path_data[i] = \
                self.bwd_path_eval(env_data=epi_path_data[i], new_Vnplus_list=Vnplus_list, prev_b_moments=prev_path_bmom_epi[i])

            # Save path data: V, H, path gain, optpar_path, H moments, conv_stat_path
            if save_vfncs:
                Vn_list_epi.append(Vnminus_list)
            H_epi.append(Hamilt)
            path_gain_list_epi.append(path_gain_list)
            new_hypparam_epi.append(hyp_param)
            path_bmom_epi.append(Hbmom)
            conv_stat_path_epi.append([Hul_norm])

            # Proceed loop: V
            Vnplus_list = Vnminus_list

        # Reverse episode data (Obtained backward)
        Vn_list_epi.reverse()
        VnT_list_epi.reverse()
        H_epi.reverse()
        path_gain_list_epi.reverse()
        term_gain_list_epi.reverse()
        path_bmom_epi.reverse()
        term_bmom_epi.reverse()
        new_hypparam_epi.reverse()
        new_hypparam_term_epi.reverse()
        conv_stat_path_epi.reverse()
        conv_stat_term_epi.reverse()

        "Forward sweep - episode-wise"
        # Declare episode data variable for Forward sweep
        new_u_epi, new_l_epi, new_mT_epi = [], [], []

        delx_init = None
        # Forward path sweep for a single leg
        delx = delx_init
        for i in range(self.leg_idx[0], self.leg_idx[1]):
            # Assign path data --> Fwd single step --> Save new trajectory
            _, delx_plus, new_u, new_Lagrn = \
                self.fwd_single_step(env_data=epi_path_data[i], path_gain_list=path_gain_list_epi[i], delx_prev=delx)

            # Save path data: u, l
            new_u_epi.append(new_u)
            new_l_epi.append(new_Lagrn)

            # Proceed loop
            delx = delx_plus

        # Boundary condition computation
        delxT = delx
        new_mT, delmT = \
            self.eval_boundary_conditions_at_term_fwd(env_data=epi_term_data[0], delxT=delxT, boundary_gain_list=term_gain_list_epi[0])
        new_mT_epi.append(new_mT)

        """Output data structure"""
        # Path data structure
        cost_aug_epi = H_epi
        epi_path_solutions = [Vn_list_epi, path_gain_list_epi,
                              new_u_epi, new_l_epi, new_hypparam_epi, cost_aug_epi, conv_stat_path_epi]

        # Terminal data structure
        epi_term_solutions = [VnT_list_epi, term_gain_list_epi,
                              new_mT_epi, new_hypparam_term_epi, conv_stat_term_epi]

        epi_value_gains = [epi_path_solutions, epi_term_solutions]

        TRmoments = [path_bmom_epi, term_bmom_epi]

        return epi_value_gains, TRmoments
