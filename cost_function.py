# author: Jelly Lee
# create date: 06/13/2024
# last modification: 06/22/2024
# status: Completed on 06/22/2024

"""
This program calculate the cost difference function used for logistic growth.
"""

import numpy as np
import matplotlib.pyplot as plt
from util import check_folder

def get_alpha_2(rho, delta_K, delta_t, c_f, c_v, c_h):
    alpha_2 = - c_h * delta_K * np.exp(-rho * delta_t) - rho * (c_f + c_v * delta_K)
    return alpha_2

def get_alpha_6(rho, delta_K, delta_t, c_f, c_v, c_u, c_h):
    alpha_6 = (c_u - c_h) * delta_K * np.exp(-rho*delta_t) - rho*(c_f + c_v*delta_K)
    return alpha_6

def get_cost_difference_function(q, A, alpha, beta, rho, K_0, delta_K, delta_t, c_f, c_v, c_u, c_h):
    alpha_1 = A * alpha * np.exp(-rho*delta_t) * (1/K_0**beta - 1/(K_0+delta_K)**beta)
    alpha_2 = get_alpha_2(rho, delta_K, delta_t, c_f, c_v, c_h)
    alpha_3 = -A * alpha * np.exp(-rho*delta_t) / (K_0 + delta_K)**beta
    alpha_4 = (A*alpha + c_u) * np.exp(-rho*delta_t)
    alpha_5 = -(c_h*delta_K + c_u*K_0) * np.exp(-rho*delta_t) - rho * (c_f + c_v*delta_K)
    alpha_6 = get_alpha_6(rho, delta_K, delta_t, c_f, c_v, c_u, c_h)

    conditions = [q <= K_0, (q > K_0) & (q <= K_0 + delta_K), q > K_0 + delta_K]
    choices = [alpha_1 * q**(beta + 1) + alpha_2, alpha_3 * q**(beta + 1) + alpha_4 * q + alpha_5, alpha_6]
    #
    # if q < K_0:
    #     return alpha_1 * q**(beta + 1) + alpha_2
    # elif q < K_0 + delta_K:
    #     return alpha_3 * q**(beta + 1) + alpha_4 * q + alpha_5
    # else:
    #     return alpha_6
    return np.select(conditions, choices)

def plot_cost_difference_function(q_array, cost_difference_array, K_0, delta_K_for_plot_cost_function, title = "Cost Difference Function", save_folder = None):
    # This function takes three different values of delta_K. Need to be modified to plot more figures
    plt.figure(figsize=(12, 6))
    color_list = ['red', 'blue', 'green']
    for i in range(3):
        plt.plot(q_array, cost_difference_array[i], linestyle='-', color=color_list[i], label = r'$\Delta K = $' + f'{delta_K_for_plot_cost_function[i]:.0e}') #,  marker='o', markersize=3)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Q', fontsize=14, fontweight='bold')
    plt.ylabel('Cost (Saving)', fontsize=14, fontweight='bold')
    plt.axvline(x = K_0, color='gray', linestyle='--', label=r'$x = K_0$')
    plt.axhline(y=0, color='black', linestyle='-', linewidth = 0.7)
    for i in range (3):
        plt.axvline(x=K_0+delta_K_for_plot_cost_function[i], color=color_list[i], linestyle='--', linewidth = 0.7)
    plt.legend()

    plt.tight_layout()

    if save_folder is not None:
        check_folder(save_folder)
        plt.savefig(save_folder + 'Cost Difference Function.pdf', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    '''The following example is for plotting purposes'''
    # from variable_definition.NumericalStudy.raw import *

    from variable_definition.PHX.raw import *

    cost_difference_array = np.zeros([3, q_array.shape[0]])
    for i in range(len(delta_K_for_plot_cost_function)):
        cost_difference_array[i] = get_cost_difference_function(q_array, A, alpha, beta, rho, K_0, delta_K_for_plot_cost_function[i], delta_t, c_f, c_v, c_u, c_h)
    plot_cost_difference_function(q_array, cost_difference_array, K_0, delta_K_for_plot_cost_function, title = "Cost Difference Function for Numerical Study", save_folder = './results/aux_plot/PHX/')


