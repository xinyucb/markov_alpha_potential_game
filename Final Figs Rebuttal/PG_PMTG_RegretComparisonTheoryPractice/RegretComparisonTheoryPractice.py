"""
    run this script to generate plots in the paper
"""
from CongestionGame import cong_game
from TG_ver import teamgame
from SeqBR import full_experiment_BR
from PolicyGradient import full_experiment_PG
from helpers import plot_accuracies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
'''
N: num of agents
runs: num of experiments
iter: the num of training epochs in each experiment
T: steps in each episodic update
samples: num of minibatch in each each episodic update
kappa: parameter used in transition 
        - set kappa to be "deterministic" if deterministic transition is used
        - set kappa to be a number such as 25, 50... if random transition with a logistic function is used
tau, tau_discouting: regularizer in SeqBR
'''

# ### Markov congestion game 
# N, runs,iters,eta,T,samples, kappa, tau, tau_discounting = 8, 2, 2000, 0.01,20,10, "deterministic", 5, 0.999
# game = cong_game(N,1,[[1,-100],[2,-100],[4,-100],[6,-100]])
# den_cong_BR, accu_cong_BR = full_experiment_BR(game, runs,iters, T,samples, kappa, tau, tau_discounting, path = "cong_BR_")
# den_cong_PG, accu_cong_PG = full_experiment_PG(game, runs,iters,eta,T,samples, kappa,  path = "cong_PG_")
# plot_accuracies(accu_cong_BR , "BR", accu_cong_PG, "PG", N, runs, path = "cong_")

# gamma_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
gamma_vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
eps_vals = [0.1]


# gamma_vals = [0.1]

# eps_vals = [0.1]

### Perturbed team game
num_runs =5
rret = {i: {j: [] for j in eps_vals} for i in gamma_vals}

for i in range(len(gamma_vals)):
    for j in range(len(eps_vals)):
        for n_avg in range(num_runs):
            N, runs,iters,eta,T,samples, kappa, gamma  = 3, 1, 1000, 0.01 , 100, 10, 1, gamma_vals[i]
            game = teamgame(N)
            game.epsilon = eps_vals[j]
        
# den_team_BR, accu_team_BR = full_experiment_BR(game, runs,iters, T,samples, kappa,tau,tau_discounting,  path="team_BR_")
            regret = full_experiment_PG(game, runs,iters,eta,T,samples, kappa, path ="team_PG_", gamma = gamma)
            print('Gamma', gamma_vals[i], 'Run_num', n_avg, 'and the regret is', regret)
            rret[gamma_vals[i]][eps_vals[j]].append(regret)
# plot_accuracies(accu_team_BR , "BR", accu_team_PG, "PG", N, runs, path = "team_")


######### Generating Plot

new_series = {i: [1/((1-i)**(9/4)) for _ in range(10)] for i in gamma_vals}

# Extracting the original rret data and reshaping it into a DataFrame
data = []
for gamma in rret:
    for eps in rret[gamma]:
        for value in rret[gamma][eps]:
            data.append({'gamma': gamma, 'eps': eps, 'value': value})

# Convert to DataFrame
df = pd.DataFrame(data)

# Plot the first set of data using seaborn's lineplot with shading for standard deviation
plt.figure(figsize=(8, 6), facecolor='white')
sns.set_style("white")  # Set seaborn style to white
sns.lineplot(x='gamma', y='value', hue='eps', data=df, ci='sd', palette='bright', marker='o')

# Plot the new series separately (mean and std dev)
means = [np.mean(new_series[gamma]) for gamma in gamma_vals]
std_devs = [np.std(new_series[gamma]) for gamma in gamma_vals]

# Plot the new series on the same plot with error bars for std deviation
plt.errorbar(gamma_vals, means, yerr=std_devs, fmt='-s', color='r', capsize=5)

# Customizing the plot
plt.xlabel('Gamma Values')
plt.ylabel('Nash Regret')
plt.yscale('log')
plt.title('Comparison of Nash regret from theory and in practice')
plt.grid(True)


# Show the plot
plt.show()