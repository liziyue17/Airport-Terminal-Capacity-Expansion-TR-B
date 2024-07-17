# author: Jelly Lee
# create date: 06/18/2024
# last modification: 06/22/2024
# status: Completed on 06/22/2024

"""
This program calculates the V function using bisection method with GIVEN F function
"""
import numpy as np
import matplotlib.pyplot as plt
from thomas_algorithm import tridiagonal_to_upper, upper_to_diag
from util import check_folder
from F_function import get_matrix_A, get_best_F_function_from_csv, get_min_q_for_positive_F

def solve_V_function(q_array, q_max, eta, sigma, rho, delta, d_last):
    """
    This program solves the V function using upper mat -> identity mat from thomas algorithm,
    with GIVEN d_last
    """
    A_mat = get_matrix_A(q_array, q_max, eta, sigma, rho, delta)
    d_vector = np.zeros_like(q_array)
    d_vector[-1] = d_last
    A_mat, d_vector = tridiagonal_to_upper(A_mat, d_vector)
    V = upper_to_diag(A_mat, d_vector)
    return V

def plot_V_function(Q_array, V_array, title):
    """
    This is only for test purposes. the plot will not be saved to any files
    """
    plt.figure(figsize=(12, 6))
    plt.plot(Q_array, V_array, linestyle='-', color='b')#,  marker='o', markersize=3)
    plt.title(title)
    plt.xlabel('Q')
    plt.ylabel('V')

    plt.tight_layout()
    plt.show()

def plot_V_F_function_old(Q_array, V_array, F_array, plot_title, opt_trigger_point = None, NPV_point = None, result_folder = None, filename = 'solve_V_F.pdf'):
    plt.figure(figsize=(12, 6))
    plt.ylim(np.max(F_array) * (-1), np.max(F_array) * 3)
    plt.plot(Q_array, V_array, linestyle='-', color='b', label = r'$V$ function')#,  marker='o', markersize=3)
    plt.plot(Q_array, F_array, linestyle='-', color='green', label = r'$F$ function')  # ,  marker='o', markersize=3)
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.xlabel(r'$Q$', fontsize=10, fontweight='bold')
    plt.ylabel('Value', fontsize=10, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.7)
    if opt_trigger_point is not None:
        plt.plot(opt_trigger_point[0], opt_trigger_point[1], marker='*', linestyle='None', markersize=15, color='red', label=r'Optimal trigger $q^*$ = ' + f'{opt_trigger_point[0]:.1e}')
    if NPV_point is not None:
        plt.plot(NPV_point[0], NPV_point[1], marker='o', linestyle='None', markersize=10, color='red', label=r'NPV solution $q_{\text{NPV}}$ = ' + f'{NPV_point[0]:.1e}')

    plt.legend(loc='best')

    plt.tight_layout()
    if result_folder is not None and filename is not None:
        check_folder(result_folder)
        plt.savefig(result_folder + filename + '.pdf', bbox_inches='tight')
    plt.show()

def plot_V_F_function(Q_array, V_array, F_array, plot_title, opt_trigger_point = None, NPV_point = None, result_folder = None, filename = 'solve_V_F.pdf', annotate_increasing = 0.5e4):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    # first plot
    ax1.plot(Q_array[V_array <= opt_trigger_point[1]], V_array[V_array <= opt_trigger_point[1]], linestyle='-', color='b', label=r'$V$ function')  # ,  marker='o', markersize=3)
    ax1.plot(Q_array, F_array, linestyle='-', color='green', label=r'$F$ function')  # ,  marker='o', markersize=3)
    ax1.set_title(r'Global View of $F$ and $V$')
    ax1.set_xlabel(r'$Q$', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Option Value / Cost', fontsize=10, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylim(np.max(F_array) * (-1), np.max(F_array) * 1.5)  # Set y-axis limits for ax1
    ax1.set_xlim(0, np.max(Q_array))  # Set y-axis limits for ax1
    if opt_trigger_point is not None:
        ax1.plot(opt_trigger_point[0], opt_trigger_point[1], marker='*', linestyle='None', markersize=15, color='red',
                 label=r'Optimal trigger $q^*$ = ' + f'{opt_trigger_point[0]:.2e}')

    ax1.legend(loc='best')

    # second plot
    ax2.plot(Q_array[V_array <= opt_trigger_point[1]], V_array[V_array <= opt_trigger_point[1]], linestyle='-', color='b', label=r'$V$ function')  # ,  marker='o', markersize=3)
    ax2.plot(Q_array, F_array, linestyle='-', color='green', label=r'$F$ function')  # ,  marker='o', markersize=3)
    ax2.set_title('Zoomed-In View for Detail')
    ax2.set_xlabel(r'$Q$', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Option Value / Cost', fontsize=10, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylim(opt_trigger_point[1] * (-0.2), opt_trigger_point[1] * 2.2)  # Set y-axis limits for ax1
    ax2.set_xlim(NPV_point[0] * 0.97, opt_trigger_point[0] * 1.02)  # Set y-axis limits for ax1
    if opt_trigger_point is not None:
        ax2.plot(opt_trigger_point[0], opt_trigger_point[1], marker='*', linestyle='None', markersize=15, color='red',
                 label=r'Optimal trigger $q^*$ = ' + f'{opt_trigger_point[0]:.2e}')
        ylim = ax2.get_ylim()
        ymin = (0 - ylim[0]) / (ylim[1] - ylim[0])  # Normalize ymin to be within the plot range
        ymax = (opt_trigger_point[1] - ylim[0]) / (ylim[1] - ylim[0])  # Normalize ymax to be within the plot range
        ax2.axvline(x=opt_trigger_point[0], ymin=ymin, ymax=ymax, color='black', linestyle='--', linewidth=1.2)

        # Calculate the midpoint for text placement
        midpoint_y = opt_trigger_point[1] / 2

        # Add text and arrows
        ax2.annotate('Wait', xy=(opt_trigger_point[0], midpoint_y),
                     xytext=(opt_trigger_point[0] - annotate_increasing, midpoint_y),  # Increase space on the left
                     fontsize=18, ha='right', va='center',fontname='Times New Roman',
                     arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))

        ax2.annotate('Invest', xy=(opt_trigger_point[0], midpoint_y),
                     xytext=(opt_trigger_point[0] + annotate_increasing, midpoint_y),  # Increase space on the right
                     fontsize=18, ha='left', va='center',fontname='Times New Roman',
                     arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))

    if NPV_point is not None:
        ax2.plot(NPV_point[0], NPV_point[1], marker='o', linestyle='None', markersize=10, color='red',
                 label=r'NPV solution $q_{\text{NPV}}$ = ' + f'{NPV_point[0]:.2e}')
    ax2.legend(loc='best')


    fig.suptitle(plot_title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if result_folder is not None and filename is not None:
        check_folder(result_folder)
        plt.savefig(result_folder + filename, bbox_inches='tight')
    plt.show()

def solve_optimal_trigger(Q_array, F_array, opt_delta_K_array, q_max, eta, sigma, rho, delta, d_last_min = 1e20,
                          d_last_max = 1e33, failure_guess_multiplier = 1e2, eps = 1, max_iters = 1000):
    # firstly, if F is always negative, no need to do investment
    if np.max(F_array) <= 0:
        print("F function is negative!")
        return None, None, None, None, None, None, -1
    # verify last and guess
    ## verify d_last_max
    while True:
        V_array = solve_V_function(Q_array, q_max, eta, sigma, rho, delta, d_last_max)
        if np.all(V_array >= F_array):
            break
        else:
            d_last_max = d_last_max * failure_guess_multiplier
        print("current d_last_max = ", d_last_max)
    ## verify d_last_min
    while True:
        V_array = solve_V_function(Q_array, q_max, eta, sigma, rho, delta, d_last_min)
        if not np.all(V_array >= F_array):
            break
        else:
            d_last_min = d_last_min / failure_guess_multiplier
        print("current d_last_min = ", d_last_min)

    # Solve opt solution using Bisection method
    for i in range(max_iters):
        d_last_medium = (d_last_max + d_last_min) / 2
        V_array = solve_V_function(Q_array, q_max, eta, sigma, rho, delta, d_last_medium)
        if np.all(V_array >= F_array):
            d_last_max = d_last_medium
        else:
            d_last_min = d_last_medium
        V_F_difference_array = np.abs(V_array - F_array)
        print("current d_last_medium = ", d_last_medium)
        if np.min(V_F_difference_array) <= eps:
            break

    opt_index = np.argmin(V_F_difference_array)
    opt_V = V_array[opt_index]
    opt_F = F_array[opt_index]
    opt_d_last = d_last_medium
    opt_trigger = Q_array[opt_index]
    opt_delta_K = opt_delta_K_array[opt_index]
    print("Solving V function finished.")
    print("opt_V = ", opt_V, "\nopt_F = ", opt_F, "\nopt_d_last = ", opt_d_last, "\nopt_trigger = ", opt_trigger, "\nopt_delta_K = ", opt_delta_K)
    return opt_V, opt_F, opt_trigger, opt_d_last, opt_delta_K, V_array, 1


if __name__ == '__main__':
    '''The following is a test example to verify that F function works well.'''
    from variable_definition.test import *

    F_array, opt_delta_K_array = get_best_F_function_from_csv(filename=cache_data_file, delta_K_array = delta_K_array, K_0 = K_0, q_max = q_max)
    opt_V, opt_F, opt_trigger, opt_d_last, opt_delta_K, V_array, status = solve_optimal_trigger(q_array, F_array, opt_delta_K_array, q_max, eta, sigma, rho, delta)

    if status != -1:
        opt_trigger_point = (opt_trigger, opt_V)
        NPV_point = (get_min_q_for_positive_F(q_array, F_array), 0)
        plot_V_F_function(q_array, V_array, F_array, plot_title, opt_trigger_point = opt_trigger_point, NPV_point = NPV_point, result_folder=result_folder)
    else:
        print("Failed due to F function non-positive.")




