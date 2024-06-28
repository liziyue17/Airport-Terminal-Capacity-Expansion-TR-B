# author: Jelly Lee
# create date: 06/27/2024
# last modification: 06/27/2024

import numpy as np

from F_function import get_best_F_function_from_csv, solve_best_F_function, plot_F_function, get_min_q_for_positive_F
from V_function import solve_optimal_trigger, plot_V_F_function_old
from util import check_folder
from main import get_F_function_result
import os

delta_t = 5
dt = 0.01
T = 50
q_0 = 0  # Can not change, or boundary condition not holds
q_max = 1e7
# eta = 0.05
eta_list = [0.03, 0.05, 0.07, 0.1, 0.13, 0.15]
sigma = 0.005
trial_times = 1000
A = 5
alpha = 3
beta = 3
rho = 0.03
K_0 = 3e6
delta_K_array = np.linspace(0, 7e6, 29)

c_f = 5e8
c_v = 100
c_u = 0.005

c_h = 0.01

delta = 5e4

num_of_grid = round((q_max - q_0 ) / delta) + 1
q_array = np.linspace(q_0, q_max, num_of_grid)


result_folder = './results/NumericalStudy/eta/'
cache_data_folder = './data/NumericalStudy/eta/'

plot_title = 'Solving Optimal Solution for Numerical Study'

if __name__ == '__main__':
    for i in range(len(eta_list)):
        print("Current trial: ", i+1, " total trials: ", len(eta_list))
        cache_data_file = cache_data_folder + 'F_mat_eta_ ' + str(eta_list[i]) + '.csv'
        check_folder(result_folder)
        check_folder(cache_data_folder)

        F_array, opt_delta_K_array = get_F_function_result(q_array, delta_K_array, delta_t, dt, T, q_max, eta_list[i], sigma, A,
                                                           alpha, beta, rho, K_0, c_f, c_v, c_u, c_h, trial_times,
                                                           delta, cache_data_folder, cache_data_file, using_cache_data=False)
        plot_F_function(q_array, F_array, label='F', title='Plot of F function')
        plot_F_function(q_array, opt_delta_K_array, label='delta K', title='Plot of relationship between Q and Delta_K')
        opt_V, opt_F, opt_trigger, opt_d_last, opt_delta_K, V_array, status = solve_optimal_trigger(q_array, F_array,
                                                                                                    opt_delta_K_array,
                                                                                                    q_max, eta_list[i], sigma,
                                                                                                    rho, delta,
                                                                                                    d_last_min=1e20,
                                                                                                    d_last_max=1e33,
                                                                                                    failure_guess_multiplier=1e2,
                                                                                                    eps=1,
                                                                                                    max_iters=1000)

        if status != -1:
            opt_trigger_point = (opt_trigger, opt_V)
            NPV_point = (get_min_q_for_positive_F(q_array, F_array), 0)
            plot_V_F_function_old(q_array, V_array, F_array, plot_title, opt_trigger_point=opt_trigger_point,
                              NPV_point=NPV_point, result_folder=result_folder, filename='solve_V_F_eta_ ' + str(eta_list[i]) + '.pdf')
        else:
            print("Failed to plot V_F function due to F function non-positive.")

