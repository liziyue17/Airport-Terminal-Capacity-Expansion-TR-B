# author: Jelly Lee
# create date: 06/22/2024
# last modification: 06/22/2024

from F_function import get_best_F_function_from_csv, solve_best_F_function, plot_F_function, get_min_q_for_positive_F, plot_relationship_of_F_and_delta_K
from V_function import solve_optimal_trigger, plot_V_F_function
from util import check_folder
import numpy as np
import os


def get_F_function_result(q_array, delta_K_array, delta_t, dt, T, q_max, eta, sigma, A, alpha, beta, rho, K_0, c_f, c_v, c_u, c_h, trial_times, delta, cache_data_folder, cache_data_file, using_cache_data = True):
    if using_cache_data and os.path.exists(cache_data_file):
        opt_F, opt_delta_K = get_best_F_function_from_csv(filename=cache_data_file, delta_K_array=delta_K_array, K_0=K_0, q_max = q_max)
    else:
        opt_F, opt_delta_K = solve_best_F_function(q_array, delta_K_array, delta_t, dt, T, q_max, eta, sigma, A, alpha, beta, rho, K_0, c_f, c_v, c_u, c_h, trial_times, delta, cache_data_folder, filename = cache_data_file)
    return opt_F, opt_delta_K

if __name__ == '__main__':
    # import parameters here
    """
    Variable definition for PHX
    """
    # from variable_definition.PHX.raw import *
    # annotate_increasing = 0.15e6
    """
    Variable definition for Numerical Study
    """
    from variable_definition.NumericalStudy.raw import *
    annotate_increasing = 0.5e4
    """
    Main Function starts here
    """
    # Create result folder
    check_folder(result_folder)
    check_folder(cache_data_folder)
    # Calculate F function and save the result
    F_array, opt_delta_K_array = get_F_function_result(q_array, delta_K_array, delta_t, dt, T, q_max, eta, sigma, A, alpha, beta, rho, K_0, c_f, c_v, c_u, c_h, trial_times, delta, cache_data_folder, cache_data_file, using_cache_data = True)
    # Plot F function and relationship between Q and delta K, this is for test purposes only!
    plot_F_function(q_array, F_array, label = 'F', title='Plot of F function')
    # plot relationship of Q and delta K
    plot_relationship_of_F_and_delta_K(q_array, opt_delta_K_array, F_array,
                                       r'Relationship Between $Q$ and $\Delta K$ for Numerical Study', result_folder=result_folder,
                                       filename='relationship_of_Q_delta_K.pdf')

    # plot_F_function(q_array, opt_delta_K_array, label = 'delta K', title='Plot of relationship between Q and Delta_K')
    # test if correct ########
    print(q_array)
    print(F_array)
    print(delta_K_array)

    # Calculate V function
    opt_V, opt_F, opt_trigger, opt_d_last, opt_delta_K, V_array, status = solve_optimal_trigger(q_array, F_array, opt_delta_K_array, q_max, eta, sigma, rho, delta, d_last_min = 1e20,
                          d_last_max = 1e33, failure_guess_multiplier = 1e2, eps = 1, max_iters = 1000)

    # Plot V and F function
    if status != -1:
        opt_trigger_point = (opt_trigger, opt_V)
        NPV_point = (get_min_q_for_positive_F(q_array, F_array), 0)
        plot_V_F_function(q_array, V_array, F_array, plot_title, opt_trigger_point=opt_trigger_point,
                          NPV_point=NPV_point, result_folder=result_folder, annotate_increasing=annotate_increasing)
    else:
        print("Failed to plot V_F function due to F function non-positive.")



