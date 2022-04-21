import numpy as np
import os

OPT_PAR_DIM = int(3)

class TrajDataPostProcessing(object):
    def __init__(self, env, save_period):
        self.env = env
        self.s_dim = env.s_dim
        self.o_dim = env.o_dim
        self.c_dim_list = env.c_dim
        self.cPineq_dim, self.cPeq_dim, self.cTineq_dim, self.cTeq_dim = self.c_dim_list
        self.cP_dim = self.cPineq_dim + self.cPeq_dim
        self.cT_dim = self.cTineq_dim + self.cTeq_dim

        self.a_dim = env.a_dim
        self.p_dim = env.p_dim
        self.optpar_dim = OPT_PAR_DIM
        self.nT = env.nT

        self.save_period = save_period
        self.init_data()

        self.path = os.getcwd().split('pdddp')[0] + 'pdddp/results/' + self.env.name + '_data/'
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)

    def init_data(self):
        # Stat history: Mean Episode value of Augmented cost, Convergence Jac norm * 2 (Path, term)
        self.epi_stat_history = np.zeros([1, 3])

        # Path data history: x, xest, y, u, r, cP, l, opP, V
        self.epi_path_data_history = np.zeros([1, self.s_dim + 2 * self.o_dim + self.a_dim + 1 + 2 * self.cP_dim + OPT_PAR_DIM])
        # Term data history: x, xest, y, r, cT, mT, opT, VT
        self.epi_term_data_history = np.zeros([1, self.s_dim + 2 * self.o_dim + 1 + 2*self.cT_dim + OPT_PAR_DIM])

        # Path gain data history: (u + cP) * (1 + x)
        self.epi_path_gain_history = np.zeros([1, (self.a_dim + self.cP_dim) * (1 + self.s_dim)])

    def stats_record(self, epi_value_gains, epi_traj, epi_misc_traj, epi_num, save_gains=False):
        """Args: epi_value_gains: CDDP solutions --->  Vlist, gain_list, path (u, l), leg (m), term (mT), master (s, n), hyperparameters, convergence stat
                 epi_traj: rollout data --->  x, xest, path (u, l), term (mT), hyperparameters, systemparameters
                 epi_misc_traj: miscellaneous data --> r, y, constraints

           Output: stat history ---> Augmented cost (or Value functions)  &  Convergence Jac norm
                   data history ---> x, y, u, r, const, Lagr, opt par, V list"""

        epi_path_data, epi_term_data = epi_traj
        epi_path_misc_data, epi_term_misc_data = epi_misc_traj

        if epi_value_gains is not None:
            epi_path_solution, epi_term_solution = epi_value_gains
        else: # Initial iteration --> No CDDP solution --> Assign null vars to path/term solutions
            epi_path_solution = [None, None, None, None, None, np.zeros([self.nT - 1, 1]), np.zeros([self.nT - 1, 1])]
            epi_term_solution = [None, None, None, None, np.zeros([1, 1])]

        "Stat history"
        # Path stat history
        Vn_list_epi, path_gain_list_epi, _, _, _, cost_aug_path_epi, conv_stat_path_epi = epi_path_solution
        epi_path_stat = np.array([[np.mean(cost_aug_path_epi), np.mean(conv_stat_path_epi)]])

        # terminal stat history
        VnT_list_epi, term_gain_list_epi, _, _, conv_stat_term_epi = epi_term_solution
        epi_term_stat = np.mean(conv_stat_term_epi).reshape([-1, 1])

        # Save episode convergence statistics
        epi_stat = np.concatenate([epi_path_stat, epi_term_stat], axis=1)
        self.epi_stat_history = np.concatenate([self.epi_stat_history, epi_stat])

        # Save FF/FB gains
        self.path_gain_list_epi = path_gain_list_epi
        self.term_gain_list_epi = term_gain_list_epi

        if epi_num % self.save_period == 0:
            "Data history"
            # Path data history
            for path_data, path_misc_data in zip(epi_path_data, epi_path_misc_data):
                x, xest, u, l, hp_path, p_mu, p_sigma, p_eps = path_data
                r, y2, cP = path_misc_data

                x_record = np.reshape(x, [1, self.s_dim])
                xest_record = np.reshape(xest, [1, self.o_dim])
                y_record = np.reshape(y2, [1, self.o_dim])
                u_record = np.reshape(u, [1, self.a_dim])
                r_record = np.reshape(r, [1, -1])

                const_record = np.reshape(cP, [1, self.cP_dim])
                lagr_record = np.reshape(l, [1, self.cP_dim])
                hyppar_record = np.reshape(hp_path, [1, -1])

                temp_data_history = np.concatenate([x_record, xest_record, y_record, u_record, r_record,
                                                    const_record, lagr_record, hyppar_record], 1).reshape([1, -1])

                self.epi_path_data_history = np.concatenate([self.epi_path_data_history, temp_data_history])

           # terminal data history
            for term_data, term_misc_data in zip(epi_term_data, epi_term_misc_data):
                xT, xT_est, mT, hp_term, p_mu, p_sigma, p_eps = term_data
                rT, yT, cT = term_misc_data

                xT_record = np.reshape(xT, [1, self.s_dim])
                xTest_record = np.reshape(xT_est, [1, self.o_dim])
                yT_record = np.reshape(yT, [1, self.o_dim])
                rT_record = np.reshape(rT, [1, 1])

                # Leg continuity constraint & Lagrangian
                constT_record = np.reshape(cT, [1, self.cT_dim])
                term_lagr_record = np.reshape(mT, [1, self.cT_dim])
                hp_term_record = np.reshape(hp_term, [1, -1])

                temp_data_history = np.concatenate([xT_record, xTest_record, yT_record, rT_record,
                                        constT_record, term_lagr_record, hp_term_record], 1).reshape([1, -1])

                self.epi_term_data_history = np.concatenate([self.epi_term_data_history, temp_data_history])

            if save_gains:
                for path_gain in path_gain_list_epi:
                    path_gain = np.concatenate([*path_gain], axis=1).T.reshape([1, -1])
                    self.epi_path_gain_history = np.concatenate([self.epi_path_gain_history, path_gain])


    def print_and_save_history(self, epi_num, save_flag=False, prefix=None):
        if prefix is None:
            prefix = str('')

        np.set_printoptions(precision=4)
        print('| Episode ', '| Cost ', '| Conv ')
        print(epi_num,
              np.array2string(self.epi_stat_history[-1, :2], formatter={'float_kind': lambda x: "    %.4f" % x}))

        np.savetxt(self.path + prefix + '_stat_history.txt', self.epi_stat_history, newline='\n')
        if epi_num % self.save_period == 0 or save_flag:
            np.savetxt(self.path + prefix + '_path_data_history.txt', self.epi_path_data_history, newline='\n')
            np.savetxt(self.path + prefix + '_term_data_history.txt', self.epi_term_data_history, newline='\n')

        if self.epi_path_gain_history is not None:
            np.savetxt(self.path + prefix + '_path_gain.txt', self.epi_path_gain_history, newline='\n')