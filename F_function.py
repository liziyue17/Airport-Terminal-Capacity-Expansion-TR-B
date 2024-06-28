# author: Jelly Lee
# create date: 06/17/2024
# last modification: 06/27/2024
# status: Completed on 06/22/2024, add numerical integral on 06/26/2024, fix bug on 06/27/2024

"""
This program calculates the F function using the thomas algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from StochasticProcessSimulation import Logistic_Demand_After_Delta_t, Logistic_Demand_Simulation, Wiener_Simulation
from cost_function import get_cost_difference_function, get_alpha_2, get_alpha_6
from thomas_algorithm import tridiagonal_to_upper, upper_to_diag
from util import check_folder

def delta_C_after_delta_t(delta_t, dt, q_0, q_max, eta, sigma, A, alpha, beta, rho, K_0, delta_K, c_f, c_v, c_u, c_h,
                          trial_times=1000):
    Q_results = Logistic_Demand_After_Delta_t(delta_t=delta_t, dt=dt, q_0=q_0, q_max=q_max, eta=eta, sigma=sigma,
                                              trial_times=trial_times)
    cost = get_cost_difference_function(Q_results, A, alpha, beta, rho, K_0, delta_K, delta_t, c_f, c_v, c_u, c_h)
    return np.mean(cost)


def get_delta_C_vector(delta_t, dt, q_array, q_max, eta, sigma, A, alpha, beta, rho, K_0, delta_K, c_f, c_v, c_u, c_h,
                       trial_times=1000):
    delta_C_vector = np.zeros_like(q_array)
    for i in range(delta_C_vector.shape[0]):
        delta_C_vector[i] = delta_C_after_delta_t(delta_t, dt, q_array[i], q_max, eta, sigma, A, alpha, beta, rho, K_0,
                                                  delta_K, c_f, c_v, c_u, c_h, trial_times)
    return delta_C_vector


'''
    F satisfies the following conditions:
    A F = d
    where A = 
    1
    a_1  b_1  c_1
         a_2  b_2  c_2
         ......
                       a_{n-2}  b_{n-2}  c_{n-2}
                                         1
    F = (F_0, F_1, F_2, ..., F_{n-2}, F_{n-1})
    d = (alpha_2/rho, d_1, d_2, ..., d_{n-2}, alpha_6/rho)
    a_i = sigma^2*q_i^2 + eta*sigma*q_i*(1-q_i/q_max)
    b_i = -2 * (sigma^2*q_i^2 + delta^2 * rho)
    c_i = sigma^2*q_i^2 - eta*sigma*q_i*(1-q_i/q_max)
    d_i = -2 * delta^2 * delta_C_after_delta_t_i
'''


def get_matrix_A(q_array, q_max, eta, sigma, rho, delta=1):
    # Step 1: calculate a_i, b_i, c_i
    a = sigma ** 2 * q_array ** 2 - eta * sigma * q_array * (1 - q_array / q_max)
    b = -2 * (sigma ** 2 * q_array ** 2 + delta ** 2 * rho)
    c = sigma ** 2 * q_array ** 2 + eta * sigma * q_array * (1 - q_array / q_max)
    matrix_dim = q_array.shape[0]
    A_mat = np.zeros((matrix_dim, matrix_dim))
    for i in range(matrix_dim):
        if i == 0 or i == matrix_dim - 1:
            A_mat[i, i] = 1
        else:
            A_mat[i, i - 1] = a[i]
            A_mat[i, i] = b[i]
            A_mat[i, i + 1] = c[i]
    return A_mat


def get_vector_d(delta_C_vector, delta, d_0, d_last):
    d = -2 * delta ** 2 * delta_C_vector
    d[0] = d_0
    d[-1] = d_last
    return d


def solve_F_function(q_array, delta_t, dt, T, q_max, eta, sigma, A, alpha, beta, rho, K_0, delta_K, c_f, c_v, c_u, c_h,
                     trial_times, delta):
    """
    This function returns a 1-d array F(q, delta K), here delta K is given
    """
    # matrices
    A_mat = get_matrix_A(q_array, q_max, eta, sigma, rho, delta)
    d_0 = get_alpha_2(rho, delta_K, delta_t, c_f, c_v, c_h) / rho
    # d_last = g
    d_last = expectation_of_F(q_max, q_max, T, dt, eta, sigma, A, alpha, beta, rho, K_0, delta_K, delta_t, c_f, c_v, c_u, c_h, trial_times = trial_times)


    delta_C_vector = get_delta_C_vector(delta_t, dt, q_array, q_max, eta, sigma, A, alpha, beta, rho, K_0, delta_K, c_f,
                                        c_v, c_u, c_h, trial_times)
    d_vector = get_vector_d(delta_C_vector, delta, d_0, d_last)

    A_mat, d_vector = tridiagonal_to_upper(A_mat, d_vector)
    F = upper_to_diag(A_mat, d_vector)
    return F


def solve_best_F_function(q_array, delta_K_array, delta_t, dt, T, q_max, eta, sigma, A, alpha, beta, rho, K_0, c_f, c_v,
                          c_u, c_h, trial_times, delta, folder, filename='F_mat.csv'):
    """
    This function saves a 2-d array F(q, delta K), for q_array and delta_K_array,
    then output the best 1-d array F function F(q) = max_{delta K} F(q, delta K(q)), which is a function of q,
    AND the corresponding 1-d array delta K for the corresponding q.
    Note: this function takes longer time than other functions, so the following function can use the cache result of the
    saved 2-d array F if this has not been done before.
    """
    F_mat = np.zeros((delta_K_array.shape[0], q_array.shape[0]))
    print("Now solving the best F function with iteration over delta K")
    for i in range(len(delta_K_array)):
        print("Current iteration: ", i + 1, ", total iteration: ", len(delta_K_array))
        print("current delta K: ", delta_K_array[i])
        F_mat[i] = solve_F_function(q_array, delta_t, dt, T, q_max, eta, sigma, A, alpha, beta, rho, K_0, delta_K_array[i],
                                    c_f, c_v, c_u, c_h, trial_times, delta)
        # print("F function result: ", F_mat[i])
    check_folder(folder)
    np.savetxt(filename, F_mat, delimiter=',')
    opt_F, opt_delta_K = get_best_F_function_from_F_mat(F_mat, delta_K_array, K_0, q_max)
    return opt_F, opt_delta_K


def get_best_F_function_from_csv(filename, delta_K_array, K_0, q_max):
    """
    This function saves did the same thing as the previous one, but using the cache csv data of 2-d function F(q, delta_K)
    """
    F_mat = np.loadtxt(filename, delimiter=',')
    opt_F, opt_delta_K = get_best_F_function_from_F_mat(F_mat, delta_K_array, K_0, q_max)
    return opt_F, opt_delta_K


# Added on 06/23/2024
def get_best_F_function_from_F_mat(F_mat, delta_K_array, K_0, q_max):
    # create a mask such that the condition K_0 + delta_K < Q_max is violated
    mask_matrix = np.zeros_like(F_mat, dtype=int)
    mask_matrix[delta_K_array + K_0 > q_max, :] = 1
    # apply the mask matrix
    F_mat[mask_matrix] = -np.inf

    opt_F = np.max(F_mat, axis=0)
    opt_F_index = np.argmax(F_mat, axis=0)
    opt_delta_K = delta_K_array[opt_F_index]
    return opt_F, opt_delta_K




def get_min_q_for_positive_F(Q_array, F_array):
    for i in range(len(Q_array)):
        if F_array[i] >= 0:
            if i == 0:
                return Q_array[i]
            else:
                x1 = Q_array[i-1]
                x2 = Q_array[i]
                y1 = F_array[i-1]
                y2 = F_array[i]
                # y = kx + b
                k = (y2-y1) / (x2-x1)
                b = y1 - k * x1
                incercept_on_x = -b / k
                return incercept_on_x#(Q_array[i-1] + Q_array[i]) / 2
    print("Function get_min_q_for_positive_F failed!")
    return -1
    
    
def plot_F_function(Q_array, y_array, label, title):
    """
    This is only for test purposes. the plot will not be saved to any files
    """
    plt.figure(figsize=(12, 6))
    plt.plot(Q_array, y_array, linestyle='-', color='b',  marker='o', markersize=3)
    plt.title(title)
    plt.xlabel('Q')
    plt.ylabel(label)

    plt.tight_layout()
    plt.show()


def numerical_integral_to_calculate_F_function(Q_trajectory_array, t_array, A, alpha, beta, rho, K_0, delta_K, delta_t, c_f, c_v, c_u, c_h):
    """
    Numerically calculate the integral F
    :param Q_trajectory_array: the trajectory for a sufficiently long time T. increment in time is dt.
    :param t_array: the same size of Q_trajectory_array, reflecting Q(t)
    :param delta_t: construction time. should be a multiplier of dt
    :return: the numerical integral of ONE trajectory, a real number
    """
    delta_C_array = get_cost_difference_function(Q_trajectory_array, A, alpha, beta, rho, K_0, delta_K, delta_t, c_f, c_v, c_u, c_h)
    # print(delta_C_array)
    time_factor_array = np.exp(-rho * t_array)
    return np.sum(delta_C_array * time_factor_array)


def expectation_of_F(Q, q_max, T, dt, eta, sigma, A, alpha, beta, rho, K_0, delta_K, delta_t, c_f, c_v, c_u, c_h, trial_times = 1000, random_seed = 42):
    if random_seed is not None:
        np.random.seed(random_seed)
    F_results_array = np.zeros(trial_times)
    for i in range(trial_times):
        # print("Iteration ", i + 1, " total iteration: ", trial_times)
        # generate q(t) array
        increment_sample, W_sample, t_array = Wiener_Simulation(T = T, dt = dt, random_state = None)
        Q_sample = Logistic_Demand_Simulation(increment_sample, dt, Q, q_max, eta, sigma)
        # get Q(t+delta_t) after the construction
        Q_trajectory_array = Q_sample[t_array >= delta_t]
        t_array_new = t_array[t_array >= delta_t] # time span after delta_t
        t_array_new = t_array_new - delta_t # starting time is zero
        F_results_array[i] = numerical_integral_to_calculate_F_function(Q_trajectory_array, t_array_new, A, alpha, beta, rho, K_0, delta_K, delta_t, c_f, c_v, c_u, c_h)
    return np.mean(F_results_array)

if __name__ == '__main__':
    '''The following is a test example to verify that F function works well.'''
    # from variable_definition.NumericalStudy.raw import *
    from variable_definition.PHX.raw import *


    # opt_F, opt_delta_K = solve_best_F_function(q_array, delta_K_array, delta_t, dt, T, q_max, eta, sigma, A, alpha,
    #                                            beta, rho, K_0, c_f, c_v, c_u, c_h, trial_times, delta, folder = cache_data_folder,
    #                                            filename=cache_data_file)
    # opt_F, opt_delta_K = get_best_F_function_from_csv(filename='./data/NumericalStudy/F_mat_raw.csv', delta_K_array = delta_K_array, K_0 = K_0, q_max = q_max)

    opt_F, opt_delta_K = get_best_F_function_from_csv(filename='./data/PHX/F_mat_raw.csv',
                                                      delta_K_array=delta_K_array, K_0=K_0, q_max=q_max)

    plot_F_function(q_array, opt_F, label = 'F', title='Plot of F function')
    plot_F_function(q_array, opt_delta_K, label = 'delta K', title='Plot of relationship between Q and Delta_K')
    print("max F value: ", np.max(opt_F))
    # F = solve_F_function(q_array, delta_t, dt, T, q_max, eta, sigma, A, alpha, beta, rho, K_0, 1e7, c_f, c_v, c_u, c_h,
    #                  trial_times, delta)
    # plot_F_function(q_array, F, label='F', title='Plot of F function GIVEN delta K')
    # d_last = expectation_of_F(q_max, q_max, T, dt, eta, sigma, A, alpha, beta, rho, K_0, 5e7, delta_t, c_f, c_v, c_u, c_h, trial_times = 1000, random_seed = 42)
    # print(d_last)


