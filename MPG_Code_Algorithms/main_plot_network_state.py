from congestion_games import *
import matplotlib.pyplot as plt
import itertools
import numpy as np
import copy
import statistics
import seaborn as sns; sns.set()
from time import process_time
import tqdm
import os

"""
The network has same state
run this script to generate plots
"""


myp_start = process_time()


# Define the states and some necessary info
N = 8 #number of agents
harm = - 100 * N # pentalty for being in bad state

safe_state = CongGame(N,1,[[1,0],[2,0],[4,0],[6,0]])
bad_state = CongGame(N,1,[[1,-100],[2,-100],[4,-100],[6,-100]])
state_dic = {0: safe_state, 1: bad_state}

M = safe_state.num_actions 
D = safe_state.m #number facilities
S = 2

# Dictionary to store the action profiles and rewards to
selected_profiles = {}

# Dictionary associating each action (value) to an integer (key)
act_dic = {}
counter = 0
for act in safe_state.actions:
	act_dic[counter] = act 
	counter += 1

def projection_simplex_sort(v, z=1):
	# Courtesy: EdwardRaff/projection_simplex.py
    if v.sum() == z and np.alltrue(v >= 0):
        return v
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w



def logistic(w, kappa, C, N):
    """
    Calculates the logistic function of x.
    """
    return 1 / (1 + np.exp(-kappa * (w/N - C)))

def bernoulli(p):
    """
    Samples a Bernoulli random variable with probability of success p.
    """
    if np.random.rand() < p:
        return 1
    else:
        return 0
    

'''
def get_next_state(state, actions):
    """deterministic transition"""
    acts_from_ints = [act_dic[i] for i in actions]
    density = state_dic[state].get_counts(acts_from_ints)
    max_density = max(density)

    if state == 0 and max_density > N/2 or state == 1 and max_density > N/4:
      # if state == 0 and max_density > N/2 and np.random.uniform() > 0.2 or state == 1 and max_density > N/4 and np.random.uniform() > 0.1:
        return 1
    return 0
'''

def sample_next_state(state, actions, kappa):
    """sample next state following a logistic function"""

    acts_from_ints = [act_dic[i] for i in actions]
    density = state_dic[state].get_counts(acts_from_ints)
    max_density = max(density)

    if kappa == "deterministic":
        if state == 0 and max_density > N/2 or state == 1 and max_density > N/4:
        # if state == 0 and max_density > N/2 and np.random.uniform() > 0.2 or state == 1 and max_density > N/4 and np.random.uniform() > 0.1:
            return 1
        return 0
    else:
        if state == 0:
            C = 1/2 + 1/(2*N)
        else:
            C = 1/4 + 1/(2*N)
        p = logistic(max_density, kappa, C, N)
        return bernoulli(p)


def pick_action(prob_dist):
    # np.random.choice(range(len(prob_dist)), 1, p = prob_dist)[0]
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action[0]


def value_function(policy, gamma, T,samples, kappa):
    """
    O(num_samples * S * T) 
    get value function by generating trajectories and calculating the rewards
    Vi(s) = sum_{t<T} gamma^t r(t)
    """
    value_fun = {(s,i):0 for s in range(S) for i in range(N)}
    for k in range(samples):
        for state in range(S):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]  # N: num of players
                q = tuple(actions+[curr_state])
                # setdefault(key, value): if key exists in dic, return its original value in dic. Else, add this new key and value into dic 
                rewards = selected_profiles.setdefault(q, get_reward(state_dic[curr_state], [act_dic[i] for i in actions]))                  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]
                curr_state = sample_next_state(curr_state, actions, kappa)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(agent, state, action, policy, gamma, value_fun, samples, kappa):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        actions[agent] = action
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q,get_reward(state_dic[state], [act_dic[i] for i in actions]))
        tot_reward += rewards[agent] + gamma*value_fun[sample_next_state(state, actions, kappa), agent]
    return (tot_reward / samples)

def policy_accuracy(policy_pi, policy_star):
    total_dif = N * [0]
    for agent in range(N):
        for state in range(S):
            total_dif[agent] += np.sum(np.abs((policy_pi[state, agent] - policy_star[state, agent])))
	  # total_dif[agent] += np.sqrt(np.sum((policy_pi[state, agent] - policy_star[state, agent])**2))
    return np.sum(total_dif) / N

def policy_gradient(mu, max_iters, gamma, eta, T, samples, kappa):

    policy = {(s,i): [1/M]*M for s in range(S) for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in tqdm.tqdm(range(max_iters)):
        eta_ = 0.99**t * eta
        b_dist = M * [1]
            
        grads = np.zeros((N, S, M))
        value_fun = value_function(policy, gamma, T, samples, kappa)
	
        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads[agent, st, act] = b_dist[st] * Q_function(agent, st, act, policy, gamma, value_fun, samples, kappa)

        for agent in range(N):
            for st in range(S):
                policy[st, agent] = projection_simplex_sort(np.add(policy[st, agent], eta_ * grads[agent,st]), z=1)
        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(policy_hist[t], policy_hist[t-1]) < 10e-16:
            return policy_hist

    return policy_hist


def get_accuracies(policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies

def full_experiment(runs,iters,eta,T,samples, kappa):
    if not os.path.exists("network_state"):
        os.makedirs("network_state")
    if not os.path.exists("network_state/kappa_"+str(kappa)):
        os.makedirs("network_state/kappa_"+str(kappa)) 

    densities = np.zeros((S,M))

    raw_accuracies = []
    for k in  tqdm.tqdm(range(runs)):
        policy_hist = policy_gradient([0.5, 0.5],iters,0.99,eta,T,samples, kappa)
        raw_accuracies.append(get_accuracies(policy_hist))

        converged_policy = policy_hist[-1]
        for i in range(N):
            for s in range(S):
                densities[s] += converged_policy[s,i]

    densities = densities / runs

    plot_accuracies = np.array(list(itertools.zip_longest(*raw_accuracies, fillvalue=np.nan))).T
    clrs = sns.color_palette("husl", 3)
    piters = list(range(plot_accuracies.shape[1]))

    fig2 = plt.figure(figsize=(6,4))
    for i in range(len(plot_accuracies)):
        plt.plot(piters, plot_accuracies[i])
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N, runs,eta))
    plt.show()
    fig2.savefig('network_state/kappa_'+str(kappa)+'/individual_runs_n{}.png'.format(N),bbox_inches='tight')
    
    plot_accuracies = np.nan_to_num(plot_accuracies)
    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))
    
    fig1 = plt.figure(figsize=(6,4))
    ax = sns.lineplot( pmean, color = clrs[0],label= 'Mean L1-accuracy')
    ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[0],label="1-standard deviation")
    ax.legend()
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N, runs,eta))
    plt.show()
    fig1.savefig('network_state/kappa_'+str(kappa)+'/avg_runs_n{}.png'.format(N),bbox_inches='tight')
    

    fig3, ax = plt.subplots()
    index = np.arange(D)
    bar_width = 0.35
    opacity = 1

    rects1 = plt.bar(index, densities[0], bar_width,
    alpha= .7 * opacity,
    color='b',
    label='Safe state')

    rects2 = plt.bar(index + bar_width, densities[1], bar_width,
    alpha= opacity,
    color='r',
    label='Distancing state')

    plt.gca().set(xlabel='Facility',ylabel='Average number of agents', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N,runs,eta))
    plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
    plt.legend()
    fig3.savefig('network_state/kappa_'+str(kappa)+'/facilities_n{}.png'.format(N),bbox_inches='tight')
    plt.show()

    return fig1, fig2, fig3


# kappa is the parameter in logisitic function.
# If you want deterministic transition, set kappa="deterministic"
runs,iters,eta,T,samples, kappa = 5,200,0.005,20,10, "deterministic"
full_experiment(runs,iters,eta,T,samples, kappa)


myp_end = process_time()
elapsed_time = myp_end - myp_start 
print(elapsed_time)
