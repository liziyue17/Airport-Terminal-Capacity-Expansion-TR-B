# author: Jelly Lee
# create date: 07/02/2024
# last modification: 07/02/2024

from F_function import get_best_F_function_from_csv, solve_best_F_function, plot_F_function, get_min_q_for_positive_F
from V_function import solve_optimal_trigger, plot_V_F_function
from util import check_folder
from main import get_F_function_result
import numpy as np
import os
import matplotlib.pyplot as plt



if __name__ == '__main__':
    # import parameters here
    from variable_definition.NumericalStudy.raw import *
    result_folder = './results/MultiPeriod/'
    cache_data_folder = './data/MultiPeriod/'
    plot_title = 'Solving Optimal Solution for Multi Period Decision Problem'
    filename = plot_title

    # Create result folder
    check_folder(result_folder)
    check_folder(cache_data_folder)

    # some parameters
    delta_K_increasing_delta = 5e4

    # Initialize parameters
    K = K_0
    flag = 0
    delta_K_set = []
    K_set = []
    trigger_demand_set = []
    plt.figure(figsize=(12, 8))
    while K < q_max and flag != 1:
        # define delta K array
        num_of_grid_delta_K = round((q_max - K) / delta_K_increasing_delta) + 1
        delta_K_array = np.linspace(0, q_max - K, num_of_grid_delta_K)

        cache_data_file = cache_data_folder + 'F_mat_' + str(len(K_set)) + '.csv'
        # Calculate F function
        F_array, opt_delta_K_array = get_F_function_result(q_array, delta_K_array, delta_t, dt, T, q_max, eta, sigma, A,
                                                           alpha, beta, rho, K, c_f, c_v, c_u, c_h, trial_times,
                                                           delta, cache_data_folder, cache_data_file,
                                                           using_cache_data=True)
        # Calculate V function
        opt_V, opt_F, opt_trigger, opt_d_last, opt_delta_K, V_array, status = solve_optimal_trigger(q_array, F_array,
                                                                                                    opt_delta_K_array,
                                                                                                    q_max, eta, sigma,
                                                                                                    rho, delta,
                                                                                                    d_last_min=1e20,
                                                                                                    d_last_max=1e33,
                                                                                                    failure_guess_multiplier=1e2,
                                                                                                    eps=1,
                                                                                                    max_iters=1000)
        if status == -1:
            flag = 1
        else:
            if opt_trigger > q_max:
                flag = 1
            else:
                # update parameters
                delta_K_set.append(opt_delta_K)
                trigger_demand_set.append(opt_trigger)
                K = K + opt_delta_K
                K_set.append(K)
                # plot figures
                if len(K_set) == 1:
                    plt.plot(q_array[V_array<=opt_V], V_array[V_array<=opt_V], linestyle='-', color='b', label=r'$V$ function')
                    plt.plot(q_array, F_array, linestyle='-', color='green', label=r'$F$ function')
                else:
                    plt.plot(q_array[V_array<=opt_V], V_array[V_array<=opt_V], linestyle='-', color='b')
                    plt.plot(q_array, F_array, linestyle='-', color='green')
                plt.plot(opt_trigger, opt_F, marker='*', linestyle='None', markersize=15,
                         color='red', label=r'$q_{{{}}}^*$ = '.format(len(K_set)) + f'{opt_trigger:.2e}')

    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.ylim(np.max(F_array) * (-0.25), np.max(F_array) * 1.6)
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.xlabel(r'$Q$', fontsize=10, fontweight='bold')
    plt.ylabel('Value', fontsize=10, fontweight='bold')
    plt.legend(loc='best')
    plt.tight_layout()
    if result_folder is not None and filename is not None:
        check_folder(result_folder)
        plt.savefig(result_folder + filename + '.pdf', bbox_inches='tight')
    plt.show()
    print("delta_K_set: ", delta_K_set)
    print("K_set: ", K_set)
    print("trigger_demand_set: ", trigger_demand_set)


