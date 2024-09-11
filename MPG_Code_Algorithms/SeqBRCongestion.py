from edgewise_game import EdgewiseCongGame, next_state, get_reward

import matplotlib.pyplot as plt
import itertools
import seaborn as sns; sns.set()
import statistics
import numpy as np
import tqdm
import numpy as np
import copy
from itertools import product
import os 

# Define the states and some necessary info
N = 8#number of agents
harm = - 100 * N # pentalty for being in bad state

all_states =  [i for i in product(range(2), repeat=4)] # all_states = [(0,0,0,0), (0,0,0,1),...]
S = len(all_states)

game = EdgewiseCongGame(N,1,[[1,-100],[2,-100],[4,-100],[6,-100]])
M = game.num_actions 
D = game.m #number facilities

# Dictionary associating each action (value) to an integer (key)
act_dic = {}
counter = 0
for act in game.actions:
	act_dic[counter] = act 
	counter += 1
selected_profiles = {}

def sample_next_state(game, state, actions, kappa):
    # state: [0, 1, 0, 1]
    acts_from_ints = [act_dic[i] for i in actions]
    density = game.get_counts(acts_from_ints)

    new_state = [0] * 4
    for j in range(D):
        new_state[j] = next_state(state[j], density[j], threshold1=N/2, threshold2=N/4, N=N, kappa=kappa)

    return tuple(new_state)

def pick_action(prob_dist):
    # np.random.choice(range(len(prob_dist)), 1, p = prob_dist)[0]
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action[0]
def entropy(policy,curr_state,player_index):
    ent = 0
    for a in range(len(policy[curr_state,player_index])):
        if policy[curr_state,player_index][a] >= 1e-6:
            ent += policy[curr_state,player_index][a]*np.log(policy[curr_state,player_index][a])

    return ent

def value_function(policy, gamma, T,samples, kappa,tau):
    """
    O(num_samples * S * T) 
    get value function by generating trajectories and calculating the rewards
    Vi(s) = sum_{t<T} gamma^t r(t)
    """
    value_fun = {(s,i):0 for s in all_states for i in range(N)}  ### change here | S before means num of states
    for k in range(samples):
        for state in all_states: ### change here | S before means num of states
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]
                q = tuple(actions+[curr_state])
                # setdefault(key, value): if key exists in di   c, return its original value in dic. Else, add this new key and value into dic 
                rewards = selected_profiles.setdefault(q, get_reward(game, [act_dic[i] for i in actions], curr_state, kappa))                  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i] - tau*(gamma**t)*entropy(policy,curr_state,i)
                curr_state = sample_next_state(game, curr_state, actions, kappa)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(agent, state, action, policy, gamma, value_fun, samples, kappa,tau,actset):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        if actset ==1:
            actions[agent] = action
            
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q, get_reward(game, [act_dic[i] for i in actions], state, kappa))                  
        tot_reward += rewards[agent] - tau*entropy(policy,state,agent) + gamma*value_fun[sample_next_state(game, state, actions, kappa), agent]
    return (tot_reward / samples)


def Q_function_one_step_policy(agent, state, policy, policy_one_step , gamma, value_fun, samples, kappa,tau):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q, get_reward(game, [act_dic[i] for i in actions], state, kappa))                  
        tot_reward += rewards[agent] - tau*entropy(policy,state,agent)- tau*entropy(policy_one_step,state,agent) + gamma*value_fun[sample_next_state(game, state, actions, kappa), agent]
    return (tot_reward / samples)

def policy_accuracy(policy_pi, policy_star):
    total_dif = N * [0]
    for agent in range(N):
        for state in all_states:
            total_dif[agent] += np.sum(np.abs((np.array(policy_pi[state, agent]) - np.array(policy_star[state, agent]))))
	  # total_dif[agent] += np.sqrt(np.sum((policy_pi[state, agent] - policy_star[state, agent])**2))
    return np.sum(total_dif) / N

def sequential_br(max_iters, gamma, eta, T, samples,kappa,tau):

    policy = {(s,i): [1/M]*M for s in all_states for i in range(N)}
    policy_exp = {(s,i): [1/M]*M for s in all_states for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in tqdm.tqdm(range(max_iters)):
        b_dist = M * [1]
            
        grads_exp = np.zeros((N, S, M))
        grads_diff = np.zeros((N,S))
        grads = np.zeros((N, S, M))
        value_fun = value_function(policy, gamma, T, samples, kappa,tau)
	
        # Computing the possible improvement for all players 
        for agent in range(N):
            for s in range(S):
                for act in range(M):
                    st = all_states[s]
                    grads_exp[agent, s, act] =  Q_function(agent, st, act, policy, gamma, value_fun, samples, kappa, tau,1)
        
        for agent in range(N):
            for s in range(S):
                st = all_states[s]
                sum_s_agent = 0
                max_val = np.max(grads_exp[agent,s])
                for a in range(M):
                    sum_s_agent += np.exp((grads_exp[agent,s,a]-max_val)/tau)
                for a in range(M):
                    policy_exp[st,agent][a]= np.exp((grads_exp[agent,s,a]-max_val)/tau)/sum_s_agent
        
        ## Finding max improvement player 
        
        for agent in range(N):
            for s in range(S):
                st = all_states[s]
                fac1 = Q_function_one_step_policy(agent, st, policy, policy_exp, gamma, value_fun, samples, kappa, tau)
                fac2 = Q_function_one_step_policy(agent, st, policy, policy, gamma, value_fun, samples, kappa, tau)
                grads_diff[agent, s] =  fac1-fac2
                                            
        max_index = np.where(grads_diff==np.max(grads_diff))
        agent_max = max_index[0][0]
        s_max = max_index[1][0]
        # print('Agent update is', agent_max)
        # print('State update is', all_states[s_max])
        for agent in range(N):
            for s in range(S):
                st = all_states[s]
                sum_s_agent = 0
                
                if agent == agent_max and s == s_max and grads_diff[agent,s]>=0:
                    max_val = np.max(grads_exp[agent,s]) 
                    for a in range(M):
                        sum_s_agent += np.exp((grads_exp[agent,s,a]-max_val)/tau)
                    for a in range(M):
                        policy[st,agent][a]= np.exp((grads_exp[agent,s,a]-max_val)/tau)/sum_s_agent
        
        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(policy_hist[t], policy_hist[t-1]) < 10e-16:
      # if policy_accuracy(policy_hist[t+1], policy_hist[t]) < 10e-16: (it makes a difference, not when t=0 but from t=1 onwards.)
            return policy_hist

    return policy_hist



def get_accuracies(policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies


def full_experiment(runs,iters,eta,T,samples, kappa,tau):
    path = "edgewise_model_results_seq_br/kappa_"+str(kappa) + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    densities = np.zeros((S,M))

    raw_accuracies = []
    for k in tqdm.tqdm(range(runs)):
        policy_hist = sequential_br(iters,0.99,eta,T,samples, kappa,tau)
        raw_accuracies.append(get_accuracies(policy_hist))

        converged_policy = policy_hist[-1]
        for i in range(N):
            for s in range(S):
                st = all_states[s]
                densities[s] += converged_policy[st,i]

    densities = densities / runs


    # Plot Figure 1: trajectories of L1 accuracy
    plot_accuracies = np.array(list(itertools.zip_longest(*raw_accuracies, fillvalue=np.nan))).T
    clrs = sns.color_palette("husl", 3)
    piters = list(range(plot_accuracies.shape[1]))

    fig2 = plt.figure(figsize=(6,4))
    for i in range(len(plot_accuracies)):
        plt.plot(piters, plot_accuracies[i])
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N, runs,eta))
    plt.show()
    fig2.savefig(path + 'individual_runs_n{}.png'.format(N),bbox_inches='tight')


    # Plot Figure 2: mean and std of L1 accuracy
    plot_accuracies = np.nan_to_num(plot_accuracies)
    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))

    fig1 = plt.figure(figsize=(6,4))
    ax = sns.lineplot( pmean, color = clrs[0],label= 'Mean L1-accuracy')
    ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[0],label="1-standard deviation")
    ax.legend()
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Sequential Best Response: agents = {}, runs = {}, $\tau$ = {}'.format(N, runs,tau))
    plt.show()
    fig1.savefig(path + 'avg_runs_n{}.png'.format(N),bbox_inches='tight')


    # Plot Figure 3: Density under different states
    fig3, ax = plt.subplots()
    index = np.arange(D)
    bar_width = 0.25
    opacity = 1

    id1 = all_states.index((0,0,0,0))
    rects1 = plt.bar(index, densities[id1], bar_width,
    alpha= .7 * opacity,
    color='b',
    label= str(all_states[id1]))

    id2 =  all_states.index((0,0,0,1))
    rects2 = plt.bar(index + bar_width, densities[id2], bar_width,
    alpha= opacity,
    color='y',
    label= str(all_states[id2]))

    id3 =  all_states.index((0,1,1,0))
    rects2 = plt.bar(index + bar_width + bar_width, densities[id3], bar_width,
    alpha= opacity,
    color='r',
    label= str(all_states[id3]))

    plt.gca().set(xlabel='Facility',ylabel='Average number of agents', title='Sequential Best Response: agents = {}, runs = {}, $\tau$ = {}'.format(N,runs,tau))
    plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
    plt.legend()
    fig3.savefig(path+ 'facilities_n{}.png'.format(N),bbox_inches='tight')
    plt.show()



# kappa is the parameter in logisitic function.
# If you want deterministic transition, set kappa="deterministic"
runs,iters,eta,T,samples, kappa, tau  = 10, 200, 0.01,20,10,"deterministic", 1
full_experiment(runs,iters,eta,T,samples, kappa,tau)