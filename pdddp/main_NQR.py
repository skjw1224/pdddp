import pickle
import os

from environments.NQRconst import NQRconstEnv
from pdddp.pdddp_iterator import CDDPIterator
from pdddp.postprocessing import TrajDataPostProcessing
from pdddp.dynamics_iterator import DynamicsIterator
from pdddp.mpc_mhe_dompc import DirectCollocation

os.environ['PYTHONPATH'] = os.getcwd()

NOISE_TRAINING_INDEX = 2
TRAIN_EPISODES = 500
TEST_EPISODES = 1
SAVE_PERIOD = 20

def train(env):
    dynamics_iterator = DynamicsIterator(env, NOISE_TRAINING_INDEX, TRAIN_EPISODES)

    postprocessing = TrajDataPostProcessing(env, save_period=SAVE_PERIOD)
    cddp_iterator = CDDPIterator(env, TRAIN_EPISODES, dynamics_iterator.leg_idx)
    mpc, _ = direct_solver(env, ddp_ref=None, full_horizon=True, closed_loop=False)

    prev_TRmoments = None
    epi_solutions = None
    CDDP_count = 0

    # Train with CDDP method
    for i in range(TRAIN_EPISODES):
        controller = {'MPC': mpc} if i == 0 else {"Open-loop DDP": None}
        # controller = {'Initial': env.initial_control} if i == 0 else {"Open-loop DDP": None}
        estimator = {'MHE': None}

        # Save plant trajectory data with closed-loop run
        epi_traj, epi_misc_traj, _ = dynamics_iterator.multiple_leg_rollout(epi_solutions, prev_epi_data=None,
                                                                         controller=controller, estimator=estimator)

        save_vfncs = True if i == TRAIN_EPISODES - 1 else False
        epi_solutions, TRmoments = cddp_iterator.solve_cddp(epi_traj, prev_TRmoments, epi_num=CDDP_count, save_vfncs=save_vfncs)
        prev_TRmoments = TRmoments
        CDDP_count += 1

        # Print statistics & Save history data
        postprocessing.stats_record(epi_solutions, epi_traj, epi_misc_traj, epi_num=i)
        postprocessing.print_and_save_history(epi_num=i, prefix='train')

    print("========================Train ended==========================")
    final_solution = [epi_solutions, epi_traj]
    # Save solutions as pkl
    with open(postprocessing.path + 'final_solution.pkl', 'wb') as fw:
        pickle.dump(final_solution, fw)

def nominal_test(env):
    dynamics_iterator = DynamicsIterator(env, NOISE_TRAINING_INDEX, TRAIN_EPISODES)
    postprocessing = TrajDataPostProcessing(env, save_period=1) # Save all test episodes

    """Cddp test"""
    print("========================Test started==========================")
    # Load solutions
    with open(postprocessing.path + 'final_solution.pkl', 'rb') as fr:
        final_solution = pickle.load(fr)

    cddp_value_gains, cddp_epi_traj = final_solution
    mpc_full, _ = direct_solver(env, ddp_ref=None, full_horizon=True, closed_loop=True)

    # Nominal with CDDP policy
    controller = {'Open-loop DDP': None}
    estimator = {'MHE': None}
    for i in range(TEST_EPISODES):
        test_epi_data, test_epi_misc_data, _ = \
            dynamics_iterator.multiple_leg_rollout(cddp_value_gains, prev_epi_data=cddp_epi_traj,
                                                   controller=controller, estimator=estimator)
        # Print statistics & Save history data
        postprocessing.stats_record(cddp_value_gains, test_epi_data, test_epi_misc_data, epi_num=i)
        postprocessing.print_and_save_history(epi_num=i, save_flag=True, prefix='test')

    # Nominal with Full horizon MPC
    controller = {'MPC': mpc_full}
    estimator = {'MHE': None}
    for i in range(TEST_EPISODES):
        test_epi_data, test_epi_misc_data, _ = \
            dynamics_iterator.multiple_leg_rollout(epi_solutions=None, prev_epi_data=None,
                                                   controller=controller, estimator=estimator)
        # Print statistics & Save history data
        postprocessing.stats_record(cddp_value_gains, test_epi_data, test_epi_misc_data, epi_num=i)
        postprocessing.print_and_save_history(epi_num=i, save_flag=True, prefix='test')

import time
import copy
def plant_test(env_plant):

    postprocessing = TrajDataPostProcessing(env_plant, save_period=1)  # Save all test episodes

    """Cddp test"""
    print("========================Test started==========================")
    # Load solutions
    with open(postprocessing.path + 'final_solution.pkl', 'rb') as fr:
        final_solution = pickle.load(fr)

    cddp_value_gains, cddp_epi_traj = final_solution

    mpc, mhe = direct_solver(env_plant, ddp_ref=None, full_horizon=False, closed_loop=True)

    # Environment needs to be redefined for each case study due to reproduce same random sequence with the fixed random seed
    print('cddp_predictor_corrector')
    dynamics_iterator = DynamicsIterator(env_plant, NOISE_TRAINING_INDEX, TRAIN_EPISODES)
    cddp_iterator = CDDPIterator(env_plant, TRAIN_EPISODES, dynamics_iterator.leg_idx)
    cddp_value_gains_copy = copy.deepcopy(cddp_value_gains)

    controller = {'Predictor-corrector DDP': cddp_iterator}
    estimator = {'MHE': mhe}

    start_time = time.time()
    for i in range(TEST_EPISODES):
        test_epi_data, test_epi_misc_data, cddp_value_gains_copy = \
            dynamics_iterator.multiple_leg_rollout(epi_solutions=cddp_value_gains_copy, prev_epi_data=cddp_epi_traj,
                                                   controller=controller, estimator=estimator)
        # Print statistics & Save history data
        postprocessing.stats_record(cddp_value_gains_copy, test_epi_data, test_epi_misc_data, epi_num=i, save_gains=True)
        postprocessing.print_and_save_history(epi_num=i, save_flag=True, prefix='plant_test')
    print("cddp_predictor_corrector time :", time.time() - start_time)

    print('cddp_corrector')
    dynamics_iterator = DynamicsIterator(env_plant, NOISE_TRAINING_INDEX, TRAIN_EPISODES)

    controller = {'Corrector DDP': None}
    estimator = {'MHE': mhe}
    start_time = time.time()
    for i in range(TEST_EPISODES):
        test_epi_data, test_epi_misc_data, _ = \
            dynamics_iterator.multiple_leg_rollout(epi_solutions=cddp_value_gains, prev_epi_data=cddp_epi_traj,
                                                   controller=controller, estimator=estimator)
        # Print statistics & Save history data
        postprocessing.stats_record(cddp_value_gains, test_epi_data, test_epi_misc_data, epi_num=i, save_gains=True)
        postprocessing.print_and_save_history(epi_num=i, save_flag=True, prefix='plant_test')
    print("cddp_corrector time :", time.time() - start_time)

    print('cddp_open_loop')
    dynamics_iterator = DynamicsIterator(env_plant, NOISE_TRAINING_INDEX, TRAIN_EPISODES)
    controller = {'Open-loop DDP': None}
    estimator = {'MHE': mhe}

    start_time = time.time()
    for i in range(TEST_EPISODES):
        test_epi_data, test_epi_misc_data, _ = \
            dynamics_iterator.multiple_leg_rollout(epi_solutions=cddp_value_gains, prev_epi_data=None,
                                                   controller=controller, estimator=estimator)
        # Print statistics & Save history data
        postprocessing.stats_record(cddp_value_gains, test_epi_data, test_epi_misc_data, epi_num=i, save_gains=True)
        postprocessing.print_and_save_history(epi_num=i, save_flag=True, prefix='plant_test')
    print("cddp_open_loop time :", time.time() - start_time)

def direct_solver(env, ddp_ref, full_horizon, closed_loop, mpc_period=None):
    nmpc_mhe = DirectCollocation(env, track_ref=ddp_ref,
                                 full_horizon=full_horizon, closed_loop=closed_loop, mpc_period=mpc_period)

    mpc = nmpc_mhe.control
    mhe = nmpc_mhe.estimate
    return mpc, mhe

def main():
    env = NQRconstEnv()
    env_plant = NQRconstEnv(prob_type='plant')

    train(env)
    # nominal_test(env)
    # plant_test(env_plant)

if __name__ == '__main__':
    main()
