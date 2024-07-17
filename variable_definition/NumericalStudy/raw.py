# author: Jelly Lee
# create date: 06/26/2024
# last modification: 06/26/2024

import numpy as np

delta_t = 5
dt = 0.01
T = 50
q_0 = 0  # Can not change, or boundary condition not holds
q_max = 1e7
eta = 0.05
sigma = 0.005
trial_times = 1000
A = 5
alpha = 3
beta = 3
rho = 0.03
K_0 = 3e6
delta_K_for_plot_cost_function = [2e6, 4e6, 6e6] #This number is for plotting cost_function purpose only.
delta_K_array = np.linspace(0, 7e6, 141)
c_f = 5e8
c_v = 100
c_u = 0.005

c_h = 0.01

delta = 5e4

num_of_grid = round((q_max - q_0 ) / delta) + 1
q_array = np.linspace(q_0, q_max, num_of_grid)




result_folder = './results/NumericalStudy/'
cache_data_folder = './data/NumericalStudy/'
cache_data_file = cache_data_folder + 'F_mat_raw.csv'
plot_title = 'Solving Optimal Solution for Numerical Study'

