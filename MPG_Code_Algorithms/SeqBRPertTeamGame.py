import numpy as np
from math import comb
import tqdm
from helpers import projection_simplex_sort
import copy
import matplotlib.pyplot as plt
import itertools as it
import os
import statistics
import seaborn as sns; sns.set()
import itertools


class TeamGame:
	#inputs: num players, max facilities per player, list of linear multiplier on utility for num of players
	def __init__(self, n):
		self.n = n # num of players
		self.epsilon = 1/(10*n)
		self.actions = [ np.random.choice(2, 1)[0] for i in range(n)]
		self.num_actions = 2
		self.all_states = [0, 1]
		self.S = len(self.all_states) # num_states


	def get_counts(self, actions):
		return np.sum(actions)

	def get_public_rewards(self, actions, state):
		density = self.get_counts(actions)
		if density < self.n / 2:
			return 0, 0
		else:
			return 1, r(state)

def r(s):
    return s
        

def xi(i, s, a, n, epsilon):
	"""
	Free feel to modify this part
    """
	xi = (s ==a) * ((n+1 -i)/n)  * 5 - a * ((i+1)/n)
	return xi * epsilon

def get_reward(game, actions, state):
	agents_rewards = game.n * [0]

	density, public_reward = game.get_public_rewards(actions, state)
	if density == 0:
		return agents_rewards
	
	for i in range(game.n):
		agents_rewards[i] = public_reward + xi(i, state, actions[i], game.n, game.epsilon)
	return agents_rewards


"""
High to High: density >= n/4
        Low:  density < n/4

Low to High: density >= n/2
       Low:  dentiy < n/2
"""
def sample_next_state(game, state, actions):
    """deterministic transition"""
    density = game.get_counts(actions)
    

    if state == 0 and density >= game.n/2 or state == 1 and density >= game.n/4:
        return 1
    return 0

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

def value_function(game, policy, gamma, T, samples, tau):
    """
    O(num_samples * S * T) 
    get value function by generating trajectories and calculating the rewards
    Vi(s) = sum_{t<T} gamma^t r(t)
    """
    S= game.S
    N = game.n
    
    selected_profiles = {}
    value_fun = {(s,i): 0 for s in range(S) for i in range(N)}
    for k in range(samples):
        for state in range(S):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]  # N: num of players
                q = tuple(actions+[curr_state])
                # setdefault(key, value): if key exists in dic, return its original value in dic. Else, add this new key and value into dic 
                rewards = selected_profiles.setdefault(q, get_reward(game, actions, state))                  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]- tau*(gamma**t)*entropy(policy,curr_state,i)
                curr_state = sample_next_state(game, curr_state, actions)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(game, agent, state, action, policy, gamma, value_fun, samples,tau,actset):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    selected_profiles = {}
    N = game.n

    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        if actset ==1:
            actions[agent] = action
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q, get_reward(game, actions, state))
        tot_reward += rewards[agent] - tau*entropy(policy,state,agent)+ gamma*value_fun[sample_next_state(game, state, actions), agent]
    return (tot_reward / samples)

# def Q_function(agent, state, action, policy, gamma, value_fun, samples, kappa,tau,actset):
#     """
#     Q = r(s, ai) + gamma * V(s)
#     """
#     tot_reward = 0
#     for i in range(samples):
#         actions = [pick_action(policy[state, i]) for i in range(N)]
#         if actset ==1:
#             actions[agent] = action
            
#         q = tuple(actions+[state])
#         rewards = selected_profiles.setdefault(q, get_reward(game, [act_dic[i] for i in actions], state, kappa))                  
#         tot_reward += rewards[agent] - tau*entropy(policy,state,agent) + gamma*value_fun[sample_next_state(game, state, actions, kappa), agent]
#     return (tot_reward / samples)
def Q_function_one_step_policy(game, agent, state, policy, policy_one_step, gamma, value_fun, samples,tau):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    selected_profiles = {}
    N = game.n

    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q, get_reward(game, actions, state))
        tot_reward += rewards[agent]  - tau*entropy(policy,state,agent)- tau*entropy(policy_one_step,state,agent) + gamma*value_fun[sample_next_state(game, state, actions), agent]
    return (tot_reward / samples)


# def Q_function_one_step_policy(agent, state, policy, policy_one_step , gamma, value_fun, samples, kappa,tau):
#     """
#     Q = r(s, ai) + gamma * V(s)
#     """
#     tot_reward = 0
#     for i in range(samples):
#         actions = [pick_action(policy[state, i]) for i in range(N)]
#         q = tuple(actions+[state])
#         rewards = selected_profiles.setdefault(q, get_reward(game, [act_dic[i] for i in actions], state, kappa))                  
#         tot_reward += rewards[agent] - tau*entropy(policy,state,agent)- tau*entropy(policy_one_step,state,agent) + gamma*value_fun[sample_next_state(game, state, actions, kappa), agent]
#     return (tot_reward / samples)

def policy_accuracy(policy_pi, policy_star, N, S):
    total_dif = N * [0]
    for agent in range(N):
        for state in range(S):
            total_dif[agent] += np.sum(np.abs((np.array(policy_pi[state, agent]) - np.array(policy_star[state, agent]))))
	  # total_dif[agent] += np.sqrt(np.sum((policy_pi[state, agent] - policy_star[state, agent])**2))
    return np.sum(total_dif) / N

def sequential_br(game, max_iters, gamma, eta, T, samples,kappa,M,S,N,tau):
    
    

    policy = {(s,i): [1/M]*M for s in range(S)for i in range(N)}
    policy_exp = {(s,i): [1/M]*M for s in range(S) for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in tqdm.tqdm(range(max_iters)):
        b_dist = M * [1]
            
        grads_exp = np.zeros((N, S, M))
        grads_diff = np.zeros((N,S))
        grads = np.zeros((N, S, M))
        value_fun = value_function(game, policy, gamma, T, samples,tau)
	
        # Computing the possible improvement for all players 
        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    # (game, agent, state, action, policy, gamma, value_fun, samples,tau,actset)
                    grads_exp[agent, st, act] =  Q_function(game, agent, st, act, policy, gamma, value_fun, samples, tau,1)
        
        for agent in range(N):
            for st in range(S):
                sum_s_agent = 0
                max_val = np.max(grads_exp[agent,st])
                for a in range(M):
                    sum_s_agent += np.exp((grads_exp[agent,st,a]-max_val)/tau)
                for a in range(M):
                    policy_exp[st,agent][a]= np.exp((grads_exp[agent,st,a]-max_val)/tau)/sum_s_agent
        
        ## Finding max improvement player 
        
        for agent in range(N):
            for st in range(S):
                # (game, agent, state, action, policy, policy_one_step, gamma, value_fun, samples,tau)
                fac1 = Q_function_one_step_policy(game, agent, st, policy, policy_exp, gamma, value_fun, samples, tau)
                fac2 = Q_function_one_step_policy(game, agent, st, policy, policy, gamma, value_fun, samples, tau)
                grads_diff[agent, st] =  fac1-fac2
                                            
        max_index = np.where(grads_diff==np.max(grads_diff))
        agent_max = max_index[0][0]
        s_max = max_index[1][0]
        # print('Agent update is', agent_max)
        # print('State update is', all_states[s_max])
        for agent in range(N):
            for st in range(S):
                sum_s_agent = 0
                
                if agent == agent_max and st == s_max and grads_diff[agent,st]>=0:
                    max_val = np.max(grads_exp[agent,st]) 
                    for a in range(M):
                        sum_s_agent += np.exp((grads_exp[agent,st,a]-max_val)/tau)
                    for a in range(M):
                        policy[st,agent][a]= np.exp((grads_exp[agent,st,a]-max_val)/tau)/sum_s_agent
        
        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(policy_hist[t], policy_hist[t-1], N,S) < 10e-16:
      # if policy_accuracy(policy_hist[t+1], policy_hist[t]) < 10e-16: (it makes a difference, not when t=0 but from t=1 onwards.)
            return policy_hist

    return policy_hist

def get_accuracies(policy_hist,  N, S):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin,  N, S)
        accuracies.append(this_acc)
    return accuracies


def full_experiment(game, runs,iters,eta,T,samples, kappa,tau):
    S, N, all_states, M = game.S, game.n, game.all_states, game.num_actions
    path = "team_model_results/kappa_"+str(kappa) + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    densities = np.zeros((S,M))
    raw_accuracies = []
    for k in tqdm.tqdm(range(runs)):
        policy_hist = sequential_br(game, iters,0.99,eta,T,samples, kappa, M, S, N,tau)
        raw_accuracies.append(get_accuracies(policy_hist, N, S))

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
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Seq BR: agents = {}, runs = {}, $\tau$ = {}'.format(N, runs,tau))
    plt.show()
    fig2.savefig(path + 'individual_runs_n{}.png'.format(N),bbox_inches='tight')

    plot_accuracies = np.nan_to_num(plot_accuracies)
    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))
    
    fig1 = plt.figure(figsize=(6,4))
    ax = sns.lineplot( pmean, color = clrs[0],label= 'Mean L1-accuracy')
    ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[0],label="1-standard deviation")
    ax.legend()
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Sequential BR: agents = {}, runs = {}, $\tau$ = {}'.format(N, runs,tau))
    plt.show()
    fig1.savefig('network_state/kappa_'+str(kappa)+'/avg_runs_n{}.png'.format(N),bbox_inches='tight')
    
    fig3, ax = plt.subplots()
    index = np.arange(M)
    bar_width = 0.35
    opacity = 1
    rects1 = plt.bar(index, densities[0], bar_width, alpha= .7 * opacity, color='b', label=r'$s_L$')

    rects2 = plt.bar(index + bar_width, densities[1], bar_width, alpha= opacity, color='r', label=r'$s_H$')

    plt.gca().set(xlabel='Action',ylabel='Average number of agents', title='Sequential BR: agents = {}, runs = {}, $\tau$ = {}'.format(N,runs,tau))
    plt.xticks(index + bar_width/2, (0, 1) )
    plt.legend()
    fig3.savefig('network_state/kappa_'+str(kappa)+'/facilities_n{}.png'.format(N),bbox_inches='tight')
    plt.show()

    return fig1, fig2, fig3

def compute_best_response(game, agent, policies, gamma=0.99, epsilon=1e-6, max_iters=1000):
    """
    Computes the best response for a given agent assuming the other agents' policies are fixed.

    :param game: Game environment
    :param agent: Index of the agent for whom we are computing the best response
    :param policies: Fixed policies for all other agents
    :param gamma: Discount factor
    :param epsilon: Convergence threshold for value iteration
    :param max_iters: Maximum number of iterations for value iteration
    :return: Optimal Q-values and the corresponding best response policy for the agent
    """
    S = game.S  # Number of states
    A = game.num_actions[agent]  # Number of actions for this agent
    
    # Initialize value function and Q-function
    V = np.zeros(S)
    Q = np.zeros((S, A))
    
    for iteration in range(max_iters):
        delta = 0
        for s in range(S):
            # Backup for each action (assuming other agents' policies are fixed)
            for a in range(A):
                expected_value = 0
                for s_prime in range(S):
                    actions_others = [np.argmax(policies[s_prime, i]) for i in range(game.n) if i != agent]
                    actions_full = actions_others[:agent] + [a] + actions_others[agent:]
                    reward = game.get_rewards(s, actions_full)[agent]
                    next_state = game.sample_next_state(s, actions_full)
                    expected_value += reward + gamma * V[next_state]
                
                Q[s, a] = expected_value  # Update Q-value
                
            best_action_value = np.max(Q[s])
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value  # Update value function
        
        if delta < epsilon:  # Convergence check
            break

    # Best response policy: always choose action that maximizes Q(s, a)
    best_response_policy = np.zeros((S, A))
    for s in range(S):
        best_action = np.argmax(Q[s])
        best_response_policy[s, best_action] = 1.0  # Deterministic best response
    
    return Q, best_response_policy

def compute_expected_value(game, policy, gamma=0.99, agent=0, epsilon=1e-6, max_iters=1000):
    """
    Computes the expected discounted value for a given agent under a specific policy.

    :param game: Game environment
    :param policy: Policy for the agent (or joint policy in multi-agent setting)
    :param gamma: Discount factor
    :param agent: Index of the agent
    :param epsilon: Convergence threshold for value iteration
    :param max_iters: Maximum number of iterations
    :return: The expected discounted value for the agent
    """
    S = game.S  # Number of states
    V = np.zeros(S)  # Initialize value function

    for iteration in range(max_iters):
        delta = 0
        for s in range(S):
            expected_value = 0
            for a in range(game.num_actions[agent]):
                action_prob = policy[s, a]
                for s_prime in range(S):
                    actions_others = [np.argmax(policy[s_prime, i]) for i in range(game.n) if i != agent]
                    actions_full = actions_others[:agent] + [a] + actions_others[agent:]
                    reward = game.get_rewards(s, actions_full)[agent]
                    next_state = game.sample_next_state(s, actions_full)
                    expected_value += action_prob * (reward + gamma * V[next_state])

            delta = max(delta, np.abs(expected_value - V[s]))
            V[s] = expected_value  # Update the value function

        if delta < epsilon:  # Convergence check
            break

    return np.mean(V)  # Return the average value across all states

def compute_nash_regret(game, policies, gamma=0.99):
    """
    Computes Nash regret for the given policies.

    :param game: Game environment
    :param policies: List of policies for all agents
    :param gamma: Discount factor
    :return: Total Nash regret value
    """
    total_regret = 0.0
    for agent in range(game.n):
        # Compute the best response for this agent assuming fixed policies for others
        Q_opt, best_response_policy = compute_best_response(game, agent, policies, gamma)
        
        # Compute expected value of current policy and best response policy
        current_value = compute_expected_value(game, policies[agent], gamma, agent)
        best_response_value = compute_expected_value(game, best_response_policy, gamma, agent)
        
        # Regret is the difference between best response and current policy value
        regret = best_response_value - current_value
        total_regret += regret
    
    return total_regret


# Example usage in the experiment setup
# Example usage in the experiment setup
def full_experiment_with_regret(game, runs, iters, eta, T, samples, kappa, tau):
    S, N, all_states, M = game.S, game.n, game.all_states, game.num_actions
    path = "team_model_results/kappa_" + str(kappa) + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    raw_accuracies = []
    raw_regrets = []
    
    for k in tqdm.tqdm(range(runs)):
        policy_hist = sequential_br(game, iters, 0.99, eta, T, samples, kappa, M, S, N, tau)
        raw_accuracies.append(get_accuracies(policy_hist, N, S))

        # Compute Nash regret for the final policy
        final_policy = policy_hist[-1]
        regret = compute_nash_regret(game, final_policy, 0.99)
        raw_regrets.append(regret)

    # Convert list of regrets to an array
    raw_regrets = np.array(raw_regrets)
    avg_regret = np.mean(raw_regrets)
    
    # Output the average Nash regret
    print("Average Nash regret:", avg_regret)
    
    return avg_regret

# Run the experiment
game = TeamGame(10)
runs, iters, eta, T, samples, kappa, tau = 5, 40, 0.01, 20, 10, "deterministic", 1

full_experiment_with_regret(game, runs, iters, eta, T, samples, kappa, tau)

# Run the experiment
game = TeamGame(10)
runs,iters,eta,T,samples, kappa, tau = 5, 40, 0.01, 20,10,"deterministic",1

full_experiment_with_regret(game, runs, iters, eta, T, samples, kappa, tau)
    
      
# kappa is the parameter in logisitic function.
# If you want deterministic transition, set kappa="deterministic"
# runs,iters,eta,T,samples, kappa, tau = 20, 40, 0.01, 20,10,"deterministic",1
# game = TeamGame(10)
# full_experiment(game, runs,iters,eta,T,samples, kappa,tau)
