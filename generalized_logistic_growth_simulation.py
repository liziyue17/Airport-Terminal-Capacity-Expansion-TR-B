# author: Jelly Lee
# create date: 07/01/2024
# last modification: 07/01/2024
from StochasticProcessSimulation import Wiener_Simulation
import matplotlib.pyplot as plt
from util import check_folder

import numpy as np

def generalized_logistic_growth(m, increment_sample, dt, q_0, q_max, eta, sigma):
    Q_sample = np.zeros_like(increment_sample)
    for i in range(len(increment_sample)):
        if i == 0:
            Q_sample[i] = q_0
        else:
            Q_sample[i] = Q_sample[i-1] + Q_sample[i-1] * (eta * (1 - (Q_sample[i-1] / q_max)**m ) * dt + sigma * increment_sample[i])
    return Q_sample
if __name__ == '__main__':
    increment_sample, W_sample, time_sample = Wiener_Simulation(T = 1000, dt = 0.01, random_state = 42)
    plt.figure(figsize=(12, 6))
    m_set = [0.2, 0.5, 1, 2, 5]
    color_set = ['red','green','grey', 'blue','orange']
    for i in range(len(m_set)):
        cumulated_sample = generalized_logistic_growth(m = m_set[i], increment_sample=increment_sample, dt=0.01, q_0=1,
                                                               q_max=3000, eta=0.02, sigma=0.01)
        plt.plot(time_sample, cumulated_sample, linestyle='-', linewidth=1.5, color=color_set[i],
                 label=r'$m = $' + str(m_set[i]))  # ,  marker='o', markersize=3)
    plt.axhline(y=3000, color='black', linestyle='--')  # , label=r'$Q = Q_{\max}$')
    title = 'Simulation of Generalized Logistic Growth'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=10, fontweight='bold')
    plt.ylabel(r'$Q$', fontsize=10, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    # save_folder = None
    save_folder = './results/simulation/'
    if save_folder is not None:
        check_folder(save_folder)
        plt.savefig(save_folder + title + '.pdf', bbox_inches='tight')
    plt.show()
