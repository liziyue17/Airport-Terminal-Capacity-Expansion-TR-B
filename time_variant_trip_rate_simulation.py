# author: Jelly Lee
# create date: 07/01/2024
# last modification: 07/01/2024
from StochasticProcessSimulation import Wiener_Simulation
import matplotlib.pyplot as plt
from util import check_folder

import numpy as np

def linear_trip_rate_simulation(theta_1, theta_0, increment_sample, time_sample, dt, q_0, q_max_0, eta, sigma):
    Q_sample = np.zeros_like(increment_sample)
    q_max_array = (time_sample * theta_1 / theta_0 + 1 ) * q_max_0
    increasing_rate_array = theta_1 / (time_sample * theta_1 + theta_0)
    for i in range(len(increment_sample)):
        if i == 0:
            Q_sample[i] = q_0
        else:
            Q_sample[i] = Q_sample[i - 1] + Q_sample[i - 1] * ((increasing_rate_array[i-1] +
                        eta * (1 - Q_sample[i - 1] / q_max_array[i-1])) * dt + sigma * increment_sample[i])
    return Q_sample, q_max_array

def quadratic_trip_rate_simulation(theta_1, theta_2, theta_0, increment_sample, time_sample, dt, q_0, q_max_0, eta, sigma):
    Q_sample = np.zeros_like(increment_sample)
    q_max_array = (time_sample**2 * theta_1 / theta_0 + time_sample * theta_2 / theta_0 + 1) * q_max_0
    increasing_rate_array = (2 * theta_1 * time_sample + theta_2) / (time_sample**2 * theta_1 + time_sample * theta_2 + theta_0)
    for i in range(len(increment_sample)):
        if i == 0:
            Q_sample[i] = q_0
        else:
            Q_sample[i] = Q_sample[i - 1] + Q_sample[i - 1] * ((increasing_rate_array[i-1] +
                        eta * (1 - Q_sample[i - 1] / q_max_array[i-1])) * dt + sigma * increment_sample[i])
    return Q_sample, q_max_array

def squareroot_trip_rate_simulation(theta_1, theta_0, increment_sample, time_sample, dt, q_0, q_max_0, eta, sigma):
    Q_sample = np.zeros_like(increment_sample)
    q_max_array = (np.sqrt(time_sample) * theta_1 / theta_0 + 1) * q_max_0
    increasing_rate_array = np.ones_like(increment_sample)
    increasing_rate_array[1:] = theta_1 / (2 * (time_sample[1:] * theta_1 + np.sqrt(time_sample[1:]) * theta_0))
    for i in range(len(increment_sample)):
        if i == 0 or i == 1:
            Q_sample[i] = q_0
        else:
            Q_sample[i] = Q_sample[i - 1] + Q_sample[i - 1] * ((increasing_rate_array[i-1] +
                        eta * (1 - Q_sample[i - 1] / q_max_array[i-1])) * dt + sigma * increment_sample[i])
    return Q_sample, q_max_array

def log_trip_rate_simulation(theta_1, theta_0, increment_sample, time_sample, dt, q_0, q_max_0, eta, sigma):
    Q_sample = np.zeros_like(increment_sample)
    q_max_array = (np.log(time_sample + 1) * theta_1 / theta_0 + 1) * q_max_0
    increasing_rate_array = theta_1 / ((time_sample + 1) * (np.log(time_sample + 1) * theta_1 + theta_0))
    for i in range(len(increment_sample)):
        if i == 0 or i == 1:
            Q_sample[i] = q_0
        else:
            Q_sample[i] = Q_sample[i - 1] + Q_sample[i - 1] * ((increasing_rate_array[i-1] +
                        eta * (1 - Q_sample[i - 1] / q_max_array[i-1])) * dt + sigma * increment_sample[i])
    return Q_sample, q_max_array



def plot_simulation_curve(time_sample, cumulated_sample, q_max_array, title, save_folder = None, rate_label = r'$Q_{\max}$'):
    plt.figure(figsize=(10, 6))
    plt.plot(time_sample, cumulated_sample, linestyle='-', linewidth=1.5, color='red', label = 'Simulation curve')#,  marker='o', markersize=3)
    plt.plot(time_sample, q_max_array, linestyle='--', linewidth=1.5, color='tab:blue', label = rate_label)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=10, fontweight='bold')
    plt.ylabel(r'$Q$', fontsize=10, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    if save_folder is not None:
        check_folder(save_folder)
        plt.savefig(save_folder + title + '.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    increment_sample, W_sample, time_sample = Wiener_Simulation(T = 1000, dt = 0.01, random_state = 42)

    cumulated_sample_linear, q_max_array = linear_trip_rate_simulation(theta_1 = 1/300, theta_0 = 1, increment_sample=increment_sample,
                                                            time_sample = time_sample, dt=0.01, q_0=1, q_max_0=3000,
                                                            eta=0.02,sigma=0.01)

    plot_simulation_curve(time_sample, cumulated_sample_linear, q_max_array, "Simulation of Linear Trip Rate Coefficient",
                          save_folder = './results/simulation/', rate_label = r'$Q_{\max}(t)$ given $\xi(t) = \theta_1 t + \theta_0$')

    cumulated_sample_linear, q_max_array = quadratic_trip_rate_simulation(theta_1=1 / 300000, theta_2=0, theta_0=1,
                                                                       increment_sample=increment_sample,
                                                                       time_sample=time_sample, dt=0.01, q_0=1,
                                                                       q_max_0=3000,
                                                                       eta=0.02, sigma=0.01)

    plot_simulation_curve(time_sample, cumulated_sample_linear, q_max_array,
                          "Simulation of Quadratic Trip Rate Coefficient", save_folder = './results/simulation/', rate_label = r'$Q_{\max}(t)$ given $\xi(t) = \theta_1 t^2 + \theta_2 t + \theta_0$')

    cumulated_sample_linear, q_max_array = squareroot_trip_rate_simulation(theta_1=1 / 10, theta_0=1,
                                                                       increment_sample=increment_sample,
                                                                       time_sample=time_sample, dt=0.01, q_0=1,
                                                                       q_max_0=3000,
                                                                       eta=0.02, sigma=0.01)

    plot_simulation_curve(time_sample, cumulated_sample_linear, q_max_array,
                          "Simulation of Square Root Trip Rate Coefficient", save_folder = './results/simulation/', rate_label = r'$Q_{\max}(t)$ given $\xi(t) = \theta_1 \sqrt{t} + \theta_0$')

    cumulated_sample_linear, q_max_array = log_trip_rate_simulation(theta_1=1 / 10, theta_0=1,
                                                                       increment_sample=increment_sample,
                                                                       time_sample=time_sample, dt=0.01, q_0=1,
                                                                       q_max_0=3000,
                                                                       eta=0.02, sigma=0.01)

    plot_simulation_curve(time_sample, cumulated_sample_linear, q_max_array,
                          "Simulation of Log Trip Rate Coefficient", save_folder = './results/simulation/', rate_label = r'$Q_{\max}(t)$ given $\xi(t) = \theta_1 \log(t+1) + \theta_0$')





