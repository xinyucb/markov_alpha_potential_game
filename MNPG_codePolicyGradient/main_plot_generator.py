"""
    run this script to generate plots in the paper
"""
from CongestionGame import cong_game
from TG_ver import teamgame
from SeqBR import full_experiment_BR
from PolicyGradient import full_experiment_PG
from helpers import plot_accuracies

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

### Perturbed team game
N, runs,iters,eta,T,samples, kappa, tau, tau_discounting, gamma  = 3, 2, 1000, 0.01 , 100,10, "deterministic", 0.05, 0.9975, 0.5
game = teamgame(N)
game.epsilon = 0.1
# den_team_BR, accu_team_BR = full_experiment_BR(game, runs,iters, T,samples, kappa,tau,tau_discounting,  path="team_BR_")
regret = full_experiment_PG(game, runs,iters,eta,T,samples, kappa, path ="team_PG_", gamma = gamma)
print(regret)
# plot_accuracies(accu_team_BR , "BR", accu_team_PG, "PG", N, runs, path = "team_")
