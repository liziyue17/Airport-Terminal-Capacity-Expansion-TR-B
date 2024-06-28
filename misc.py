# author: Jelly Lee
# create date: 06/26/2024
# last modification: 06/26/2024
# status: Completed on 06/26/2024
"""
This file is to compare the F(Q_max) calculated by numerical integral and alpha_6/rho
"""
import numpy as np
from cost_function import get_alpha_6
from F_function import expectation_of_F


if __name__ == "__main__":
    # Define constants
    delta_t = 3 # construction time
    dt = 0.01 # numerical integral gap
    q_max = 6e7
    eta = 0.13
    sigma = 0.05
    A = 5
    alpha = 2
    beta = 3
    rho = 0.03
    K_0 = 3e7
    delta_K = 1.5e7
    c_f = 2e9
    c_v = 100
    c_u = 0.005
    c_h = 0.01
    T = 20 # years of calculation period, should be LARGER than delta_t because time 0 to delta_t is not calculated
    # Calculate using alpha_6/rho
    cost_direct = get_alpha_6(rho, delta_K, delta_t, c_f, c_v, c_u, c_h) / rho
    print("Calculate using alpha_6/rho: ", cost_direct)
    # Calculate using numerical integral
    cost_simulation = expectation_of_F(q_max, q_max, T, dt, eta, sigma, A, alpha, beta, rho, K_0, delta_K, delta_t, c_f, c_v, c_u, c_h, trial_times = 1000, random_seed = None)
    print("Calculate using numerical integral: ", cost_simulation)
    error = np.abs(cost_direct - cost_simulation)
    print("Error: ", error / np.abs(cost_simulation))
