# author: Jelly Lee
# create date: 06/13/2024
# last modification: 06/22/2024
# status: Completed on 06/22/2024

"""
This program simulates GBM and logistic growth.
    Wiener_Simulation: simulate wiener white noise, i.e., i.i.d. Gaussian with zero mean and s.d. given by dt
    Logistic_Demand_Simulation: simulate logistic growth
    GBM_Demand_Simulation: simulate GBM
    Logistic_Demand_After_Delta_t: simulate the demand after time delta_t for the logistic process
"""

import numpy as np
import matplotlib.pyplot as plt
from util import check_folder

def Wiener_Simulation(T, dt, random_state = None):
    """    W(t+u) - W(t) ~ Normal(0, u)    """
    if random_state:
        np.random.seed(random_state)
    N = round(T / dt)
    increment_sample = np.random.normal(loc = 0, scale = np.sqrt(dt), size = N)
    W_sample = np.cumsum(increment_sample)
    time_sample = np.arange(N) * dt
    return increment_sample, W_sample, time_sample

def Logistic_Demand_Simulation(increment_sample, dt, q_0, q_max, eta, sigma):
    Q_sample = np.zeros_like(increment_sample)
    for i in range(len(increment_sample)):
        if i == 0:
            Q_sample[i] = q_0
        else:
            Q_sample[i] = Q_sample[i-1] + Q_sample[i-1] * (eta * (1 - Q_sample[i-1] / q_max) * dt + sigma * increment_sample[i])
    return Q_sample

def Logistic_Demand_After_Delta_t(delta_t, dt, q_0, q_max, eta, sigma, trial_times = 1000):
    Q_results = np.zeros(trial_times)
    for trial in range(trial_times):
        increment_sample, _, _ = Wiener_Simulation(delta_t, dt)
        Q_sample = np.zeros_like(increment_sample)
        for i in range(len(increment_sample)):
            if i == 0:
                Q_sample[i] = q_0
            else:
                Q_sample[i] = Q_sample[i-1] + Q_sample[i-1] * (eta * (1 - Q_sample[i-1] / q_max) * dt + sigma * increment_sample[i])
        Q_results[trial] = Q_sample[-1]
    # Demand after time dt is given by taking the expectation of those results
    return Q_results  # This is a numpy array of all trial results of the demand after time delta_t


def GBM_Demand_Simulation(increment_sample, dt, q_0, eta, sigma):
    Q_sample = np.zeros_like(increment_sample)
    for i in range(len(increment_sample)):
        if i == 0:
            Q_sample[i] = q_0
        else:
            Q_sample[i] = Q_sample[i-1] + Q_sample[i-1] * (eta * dt + sigma * increment_sample[i])
    return Q_sample

def plot_simulation_curve(time_sample, cumulated_sample, title, plot_horizontal = None, save_folder = None):
    plt.figure(figsize=(12, 6))
    plt.plot(time_sample, cumulated_sample, linestyle='-', linewidth=1.5, color='tab:blue', label = 'Simulation curve')#,  marker='o', markersize=3)
    if plot_horizontal is not None:
        plt.axhline(y=plot_horizontal, color='black', linestyle='--', label=r'$Q = Q_{\max}$')
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Time', fontsize=14, fontweight='bold')
    plt.ylabel(r'$Q$', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    if save_folder is not None:
        check_folder(save_folder)
        plt.savefig(save_folder + title + '.pdf', bbox_inches='tight')
    plt.show()

def plot_gbm_and_logistic(time_sample, cumulated_sample_gbm, cumulated_sample_logistic, plot_horizontal,
                          title = 'Comparison of GBM and Logistic Growth', save_folder = None):
    plt.figure(figsize=(12, 6))
    plt.ylim(0, plot_horizontal * 3)
    plt.plot(time_sample, cumulated_sample_logistic, linestyle='-', linewidth=1, color='tab:blue',
             label='Logistic Growth')  # ,  marker='o', markersize=3)
    plt.plot(time_sample, cumulated_sample_gbm, linestyle='-', linewidth=1, color='tab:red',
             label='GBM')
    plt.axhline(y=plot_horizontal, color='black', linestyle='--')#, label=r'$Q = Q_{\max}$')

    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Time', fontsize=14, fontweight='bold')
    plt.ylabel(r'$Q$', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    if save_folder is not None:
        check_folder(save_folder)
        plt.savefig(save_folder + title + '.pdf', bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    '''The following is to simulate and plot GBM and logistic growth.'''
    # increment_sample, W_sample, time_sample = Wiener_Simulation(T = 800, dt = 0.01, random_state = 42)
    #
    # cumulated_sample_logistic = Logistic_Demand_Simulation(increment_sample=increment_sample, dt=0.01, q_0=1, q_max=3000, eta=0.02,sigma=0.01)
    # cumulated_sample_gbm = GBM_Demand_Simulation(increment_sample=increment_sample, dt=0.01, q_0=1, eta=0.02, sigma=0.01)
    #
    # # plot_simulation_curve(time_sample, cumulated_sample_logistic, "Simulation of Logistic Curve", plot_horizontal = 2000, save_folder = './results/simulation/')
    #
    # plot_gbm_and_logistic(time_sample, cumulated_sample_gbm, cumulated_sample_logistic, 3000, save_folder='./results/simulation/')

    '''The following is a test example to verify that Logistic_Demand_After_Delta_t works well.'''
    # Q = Logistic_Demand_After_Delta_t(delta_t=10, dt=0.01, q_0=100, q_max=2000, eta=0.03, sigma=0.01, trial_times=1000)
    # print(np.mean(Q))


    '''The following is the simulation for numerical experiments'''
    from variable_definition.NumericalStudy.raw import *

    plt.figure(figsize=(12, 6))
    # generate 3 samples
    color = ['red', 'blue', 'green']
    for i in range(3):
        increment_sample, W_sample, time_sample = Wiener_Simulation(300, dt)
        cumulated_sample = Logistic_Demand_Simulation(increment_sample, dt, 1e5, q_max, eta, sigma)
        plt.plot(time_sample, cumulated_sample, linestyle='-', linewidth=1, color=color[i])
    plt.axhline(y=q_max, color='black', linestyle='--', label=r'$Q = Q_{\max}$')
    plt.title('Sample Paths of Demand Growth', fontsize=18, fontweight='bold')
    plt.xlabel('Time', fontsize=14, fontweight='bold')
    plt.ylabel(r'$Q$', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    save_folder = './results/simulation/'
    check_folder(save_folder)
    plt.savefig(save_folder + 'Sample Paths of Demand Growth' + '.pdf', bbox_inches='tight')
    plt.show()



