# author: Jelly Lee
# create date: 06/22/2024
# last modification: 06/22/2024

"""
Variables defined here are for test purposes only. Iteration times are greatly simplified for debugging.
"""

import numpy as np

delta_t = 3
dt = 0.1
T = 30
q_0 = 0  # Can not change, or boundary condition not holds
q_max = 6e7
eta = 0.13
sigma = 0.055
trial_times = 1000
A = 5
alpha = 2
beta = 3
rho = 0.03
K_0 = 3e7
delta_K = 0.5e7 #This number is for plotting cost_function purpose only.
delta_K_array = np.linspace(0, 10e7, 26)
c_f = 2e9
c_v = 100
c_u = 0.005

c_h = 0.01

delta = 0.5e6

num_of_grid = round((q_max - q_0 ) / delta) + 1
q_array = np.linspace(q_0, q_max, num_of_grid)

result_folder = './results/test/'
cache_data_folder = './data/test/'
cache_data_file = cache_data_folder + 'F_mat.csv'
plot_title = '(For test only) Solving Optimal Solution'

