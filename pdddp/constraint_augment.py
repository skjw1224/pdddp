import numpy as np

GAMMA = 2

class ConstraintAug(object):
    def __init__(self, s_dim, a_dim, c_dim_list):
        self.s_dim, self.a_dim = s_dim, a_dim

        self.cPineq_dim, self.cPeq_dim, self.cTineq_dim, self.cTeq_dim = c_dim_list
        self.cP_dim = self.cPineq_dim + self.cPeq_dim
        self.cT_dim = self.cTineq_dim + self.cTeq_dim
        self.gamma = GAMMA

    def check_active_idx(self, constraint, Lagrangian, const_type):
        G, Gx, Gu, Gxx, Gxu, Guu = constraint
        eps = np.finfo(float).eps
        c = self.c  # (C, )

        # Powell-Hestenes-Rockafeller (PHR) method for inequality constraint
        # Active set: Infeasible: 0, Strictly feasible: 1, Weakly feasible: 2
        active_set = np.zeros([self.c_dim, 1])
        strf_idx = []
        wkf_idx = []
        inf_idx = []

        if const_type == 'path':
            ineq_idx_range, eq_idx_range = self.cPineq_dim, self.cPeq_dim
        elif const_type == 'terminal':
            ineq_idx_range, eq_idx_range = self.cTineq_dim, self.cTeq_dim

        # Active set based on the Lagrangian
        lambda_ = Lagrangian  # 1*C
        for i in range(ineq_idx_range):
            if G[i] <= - lambda_[i] / c:
                active_set[i] = 1
                strf_idx.append(i)
            elif lambda_[i] >= c * eps and - lambda_[i] / c <= G[i] < eps:
                active_set[i] = 2
                wkf_idx.append(i)
            else:
                inf_idx.append(i)

        for i in range(eq_idx_range):
            inf_idx.append(i + ineq_idx_range)

        return active_set, strf_idx, wkf_idx, inf_idx

    def costfcns_strf(self, strf_idx, Lagrangian):
        strf_dim = len(strf_idx)
        c = self.c
        if strf_dim != 0:
            lambda_ = Lagrangian[strf_idx, :]  # (C*1)

            # Cost Augmentation for inactive constraint
            P = - 1 / (2 * c) * lambda_.T @ lambda_
            Px = np.zeros([self.s_dim, 1])                  # (S*1)
            Pu = np.zeros([self.u_dim, 1])                  # (A*1)
            Pl = - 1 / c * lambda_                          # (C*1)
            Pxx = np.zeros([self.s_dim, self.s_dim])        # (S*S)
            Pxu = np.zeros([self.s_dim, self.u_dim])        # (S*A)
            Pxl = np.zeros([self.s_dim, strf_dim])          # (S*C)
            Puu = np.zeros([self.u_dim, self.u_dim])        # (A*A)
            Pul = np.zeros([self.u_dim, strf_dim])          # (C*A)
            Pll = - 1 / c * np.eye(strf_dim)  # (C*C)
        else:
            P, Px, Pu, Pl, Pxx, Pxu, Pxl, Puu, Pul, Pll = self.costfcns_null(strf_dim)
        return P, Px, Pu, Pl, Pxx, Pxu, Pxl, Puu, Pul, Pll

    def costfcns_wkf(self, wkf_idx, constraint, Lagrangian):
        G, Gx, Gu, Gxx, Gxu, Guu = constraint
        wkf_dim = len(wkf_idx)
        c = self.c

        if wkf_dim != 0:
            G = G[wkf_idx, :]  # (C*1)
            Gx = Gx[wkf_idx, :]  # (C*S)
            Gu = Gu[wkf_idx, :]  # (C*A)
            Gxx = Gxx[wkf_idx, :, :]  # (C*S*S)
            Gxu = Gxu[wkf_idx, :, :]  # (C*S*A)
            Guu = Guu[wkf_idx, :, :]  # (C*A*A)

            lambda_ = Lagrangian[wkf_idx, :]  # (C*1)

            # Cost Augmentation for active constraint
            P = lambda_.T @ G + c * G.T @ G - 1 / (2 * c) * lambda_.T @ lambda_   # (1*1)
            Px = Gx.T @ (2 * c * G + lambda_)  # (S*1)
            Pu = Gu.T @ (2 * c * G + lambda_)  # (A*1)
            Pl = G - 1 / c * lambda_       # (C*1)

            Gx_ = Gx[:, :, np.newaxis]  # C*S --> C*S*1
            G_x = Gx[:, np.newaxis, :]  # C*S --> C*1*S
            Gu_ = Gu[:, :, np.newaxis]  # C*A --> C*A*1
            G_u = Gu[:, np.newaxis, :]  # C*A --> C*1*A
            # dot( (1*C), (S*C*S) ) --> 1*S*S --> S*S     ////         # matmul( (C*S*1), (C*1*S) ) --> C*S*S --> S*C*S
            Pxx = np.dot((2 * c * G + lambda_).T, np.transpose(Gxx, [1, 0, 2]))[0] + 2 * c * np.sum(np.matmul(Gx_, G_x), axis=0)  # (S*S)
            Pxu = np.dot((2 * c * G + lambda_).T, np.transpose(Gxu, [1, 0, 2]))[0] + 2 * c * np.sum(np.matmul(Gx_, G_u), axis=0)  # (S*A)
            Pxl = Gx.T                                                                                                            # (S*C)
            Puu = np.dot((2 * c * G + lambda_).T, np.transpose(Guu, [1, 0, 2]))[0] + 2 * c * np.sum(np.matmul(Gu_, G_u), axis=0)  # (A*A)
            Pul = Gu.T                                                                                                            # (A*C)
            Pll = - 1 / c * np.eye(wkf_dim)                                                                                       # (C*C)
        else:
            P, Px, Pu, Pl, Pxx, Pxu, Pxl, Puu, Pul, Pll = self.costfcns_null(wkf_dim)
        return P, Px, Pu, Pl, Pxx, Pxu, Pxl, Puu, Pul, Pll

    def costfcns_inf(self, i_idx, constraint, Lagrangian):
        G, Gx, Gu, Gxx, Gxu, Guu = constraint
        i_dim = len(i_idx)
        c = self.c

        if i_dim != 0:
            G = G[i_idx, :]                        # (C*1)
            Gx = Gx[i_idx, :]                      # (C*S)
            Gu = Gu[i_idx, :]                      # (C*A)
            Gxx = Gxx[i_idx, :, :]                 # (C*S*S)
            Gxu = Gxu[i_idx, :, :]                 # (C*S*A)
            Guu = Guu[i_idx, :, :]                 # (C*A*A)

            lambda_ = Lagrangian[i_idx, :]  # (1*C)

            # Cost Augmentation for active constraint
            P = lambda_.T @ G + c / 2 * G.T @ G - 1 / (2 * c) * lambda_.T @ lambda_   # (1*1)
            Px = Gx.T @ (c * G + lambda_)  # (S*1)
            Pu = Gu.T @ (c * G + lambda_)  # (A*1)
            Pl = G - 1 / c * lambda_  # (C*1)

            Gx_ = Gx[:, :, np.newaxis]  # C*S --> C*S*1
            G_x = Gx[:, np.newaxis, :]  # C*S --> C*1*S
            Gu_ = Gu[:, :, np.newaxis]  # C*A --> C*A*1
            G_u = Gu[:, np.newaxis, :]  # C*A --> C*1*A
            # dot( (1*C), (S*C*S) ) --> 1*S*S --> S*S     ////         # matmul( (C*S*1), (C*1*S) ) --> C*S*S --> S*C*S
            Pxx = np.dot((c * G + lambda_).T, np.transpose(Gxx, [1, 0, 2]))[0] + c * np.sum(np.matmul(Gx_, G_x), axis=0)    # (S*S)
            Pxu = np.dot((c * G + lambda_).T, np.transpose(Gxu, [1, 0, 2]))[0] + c * np.sum(np.matmul(Gx_, G_u), axis=0)    # (S*A)
            Pxl = Gx.T                                                                                                      # (S*C)
            Puu = np.dot((c * G + lambda_).T, np.transpose(Guu, [1, 0, 2]))[0] + c * np.sum(np.matmul(Gu_, G_u), axis=0)    # (A*A)
            Pul = Gu.T                                                                                                      # (A*C)
            Pll = - 1 / c * np.eye(i_dim)                                                                                   # (C*C)
        else:
            P, Px, Pu, Pl, Pxx, Pxu, Pxl, Puu, Pul, Pll = self.costfcns_null(i_dim)

        return P, Px, Pu, Pl, Pxx, Pxu, Pxl, Puu, Pul, Pll

    def costfcns_null(self, c_dim):
        # When index is empty
        P, Px, Pu, Pxx, Pxu, Puu = np.zeros([1, 1]), np.zeros([self.s_dim, 1]), np.zeros([self.u_dim, 1]), \
                                   np.zeros([self.s_dim, self.s_dim]), np.zeros([self.s_dim, self.u_dim]), np.zeros([self.u_dim, self.u_dim])
        if c_dim != 0:
            Pl, Pxl, Pul, Pll = np.zeros([c_dim, 1]), np.zeros([self.s_dim, c_dim]), np.zeros([self.u_dim, c_dim]), np.zeros([c_dim, c_dim])
        else:
            Pl, Pxl, Pul, Pll = [], [], [], []
        return P, Px, Pu, Pl, Pxx, Pxu, Pxl, Puu, Pul, Pll

    def aug_constfncs(self, constraint, Lagrangian, penalty, const_type):
        """const_type: path, leg, terminal"""
        # path constraint: cP_dim, a_dim
        # terminal constraint: cT_dim, cTeq_dim (null)
        if const_type == 'path': # cP, u
            self.c_dim, self.u_dim = self.cP_dim, self.a_dim
        elif const_type == 'terminal': # cT,
            self.c_dim, self.u_dim = self.cT_dim, 0

        self.c = penalty

        if const_type == 'terminal':
            # Assign fictitious u-vars to terminal constraint --> Compatibility to aug constfncs args
            constraint = [constraint[0], constraint[1], np.zeros([self.cT_dim, 0]),
                          np.concatenate(constraint[2:]).reshape([-1, self.s_dim, self.s_dim]),
                          np.zeros([self.cT_dim, self.s_dim, 0]),
                          np.zeros([self.cT_dim, 0, 0])]
        else:
            constraint = [constraint[0], constraint[1], constraint[2],
                          np.concatenate(constraint[3:3 + self.c_dim]).reshape([-1, self.s_dim, self.s_dim]),
                          np.concatenate(constraint[3 + self.c_dim:3 + 2 * self.c_dim]).reshape([-1, self.s_dim, self.u_dim]),
                          np.concatenate(constraint[-self.c_dim:]).reshape([-1, self.u_dim, self.u_dim])]  # (C*1), (C*S), (C*A), (C*S*S), (C*S*A), (C*A*A)

        active_set, strf_idx, wkf_idx, inf_idx = self.check_active_idx(constraint, Lagrangian, const_type)
        P_strf, Px_strf, Pu_strf, Pl_strf, Pxx_strf, Pxu_strf, Pxl_strf, Puu_strf, Pul_strf, Pll_strf = self.costfcns_strf(strf_idx, Lagrangian)
        P_wkf, Px_wkf, Pu_wkf, Pl_wkf, Pxx_wkf, Pxu_wkf, Pxl_wkf, Puu_wkf, Pul_wkf, Pll_wkf = self.costfcns_wkf(wkf_idx, constraint, Lagrangian)
        P_inf, Px_inf, Pu_inf, Pl_inf, Pxx_inf, Pxu_inf, Pxl_inf, Puu_inf, Pul_inf, Pll_inf = self.costfcns_inf(inf_idx, constraint, Lagrangian)

        P = P_strf + P_wkf + P_inf
        Px = Px_strf + Px_wkf + Px_inf
        Pu = Pu_strf + Pu_wkf + Pu_inf
        Pxx = Pxx_strf + Pxx_wkf + Pxx_inf
        Pxu = Pxu_strf + Pxu_wkf + Pxu_inf
        Puu = Puu_strf + Puu_wkf + Puu_inf

        Pl = np.zeros([self.c_dim, 1])
        Pxl = np.zeros([self.s_dim, self.c_dim])
        Pul = np.zeros([self.u_dim, self.c_dim])
        Pll = np.zeros([self.c_dim, self.c_dim])

        if len(strf_idx) != 0:
            Pl[strf_idx, :] = Pl_strf
            Pxl[:, strf_idx] = Pxl_strf
            Pul[:, strf_idx] = Pul_strf
            Pll[np.ix_(strf_idx, strf_idx)] = Pll_strf  # np.ix_ --> 2d array slicing

        if len(wkf_idx) != 0:
            Pl[wkf_idx, :] = Pl_wkf
            Pxl[:, wkf_idx] = Pxl_wkf
            Pul[:, wkf_idx] = Pul_wkf
            Pll[np.ix_(wkf_idx, wkf_idx)] = Pll_wkf  # np.ix_ --> 2d array slicing

        if len(inf_idx) != 0:
            Pl[inf_idx, :] = Pl_inf
            Pxl[:, inf_idx] = Pxl_inf
            Pul[:, inf_idx] = Pul_inf
            Pll[np.ix_(inf_idx, inf_idx)] = Pll_inf
        
        aug_fcns_list = [P, Px, Pu, Pl, Pxx, Pxu, Pxl, Puu, Pul, Pll]
        return aug_fcns_list, active_set