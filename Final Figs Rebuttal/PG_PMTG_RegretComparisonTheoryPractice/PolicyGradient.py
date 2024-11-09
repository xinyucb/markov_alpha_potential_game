import matplotlib.pyplot as plt
import itertools
import seaborn as sns; sns.set()
import statistics
import numpy as np
import tqdm
import numpy as np
import copy


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




def pick_action(prob_dist):
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action[0]


def value_function(game, policy, gamma, T,samples, kappa):
    """
    O(num_samples * S * T) 
    get value function by generating trajectories and calculating the rewards
    Vi(s) = sum_{t<T} gamma^t r(t)
    """
    act_dic , N ,  all_states= game.act_dic, game.n, game.all_states
    selected_profiles = {}
    value_fun = {(s,i):0 for s in all_states for i in range(N)}  
    for k in range(samples):
        for state in all_states: 
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]
                q = tuple(actions+[curr_state])
                # setdefault(key, value): if key exists in di   c, return its original value in dic. Else, add this new key and value into dic 
                rewards = selected_profiles.setdefault(q, game.get_reward([act_dic[i] for i in actions], curr_state, kappa))                  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]
                curr_state = game.sample_next_state(curr_state, actions, kappa)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(game, agent, state, action, policy, gamma, value_fun, samples, kappa):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    act_dic , N = game.act_dic, game.n
    selected_profiles = {}
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        actions[agent] = action
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q, game.get_reward([act_dic[i] for i in actions], state, kappa))                  
        tot_reward += rewards[agent] + gamma*value_fun[game.sample_next_state( state, actions, kappa), agent]
    return (tot_reward / samples)

def policy_accuracy(game, policy_pi, policy_star):
    N ,  all_states=  game.n, game.all_states
    total_dif = N * [0]
    for agent in range(N):
        for state in all_states:
            total_dif[agent] += np.sum(np.abs((policy_pi[state, agent] - policy_star[state, agent])))
	  # total_dif[agent] += np.sqrt(np.sum((policy_pi[state, agent] - policy_star[state, agent])**2))
    return np.sum(total_dif) / N

def policy_gradient(game, max_iters, gamma, eta, T, samples,kappa):
    N ,  all_states, M, S=  game.n, game.all_states, game.m, game.S
    policy = {(s,i): [1/M]*M for s in all_states for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in tqdm.tqdm(range(max_iters)):
        eta_ = 1**t * eta
        b_dist = M * [1]
            
        grads = np.zeros((N, S, M))
        value_fun = value_function(game, policy, gamma, T, samples, kappa)
	
        for agent in range(N):
            for s in range(S):
                for act in range(M):
                    st = all_states[s]
                    grads[agent, s, act] =  Q_function(game, agent, st, act, policy, gamma, value_fun, samples, kappa)

        for agent in range(N):
            for s in range(S):
                st = all_states[s]
                policy[st, agent] = projection_simplex_sort(np.add(policy[st, agent], eta_ * grads[agent,s]), z=1)
        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(game, policy_hist[t], policy_hist[t-1]) < 10e-16:
            return policy_hist

    return policy_hist


def get_accuracies(game, policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(game, policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies


def compute_best_response(game, agent, policies, gamma=0.99, kappa = "deterministic", epsilon=9e-3, max_iters=1000, num_samples=1000):
    """
    Computes the best response for a given agent, assuming other agents' policies are fixed.

    :param game: Game environment
    :param agent: Index of the agent for whom we're computing the best response
    :param policies: Fixed policies for all agents (stochastic)
    :param gamma: Discount factor
    :param epsilon: Convergence threshold for value iteration
    :param max_iters: Maximum number of iterations
    :param num_samples: Number of samples to estimate expectations
    :return: Q-values and best response policy for the agent
    """
    S = game.S  # Number of states
    A = game.m  # Number of actions

    # Initialize value function and Q-function for the agent
    V = np.zeros(S)
    Q = np.zeros((S, A))

    for iteration in range(max_iters):
        delta = 0
        # Iterate over all states
        for s in range(S):
            # Backup the value for each action available to the agent
            for a in range(A):
                expected_value = 0
                
                # Estimate the expected value for taking action 'a' by the agent
                for _ in range(num_samples):
                    # Sample actions for the other agents based on their policies
                    actions_others = [
                        np.random.choice(
                            len(policies[(s, i)]), p=policies[(s, i)]
                        ) for i in range(game.n) if i != agent
                    ]

                    # Form the full joint action, including the agent's action 'a'
                    actions_full = actions_others[:agent] + [a] + actions_others[agent:]

                    # Get the reward for the agent from the joint action
                    reward = game.get_public_rewards(actions_full, s)[0] + game.xi(agent, s, a, game.n, game.epsilon)

                    # Sample the next state based on the joint action
                    next_state = game.sample_next_state(s, actions_full, kappa)

                    # Add discounted future value to the expected value
                    expected_value += reward + gamma * V[next_state]

                # Average the sampled values
                Q[s, a] = expected_value / num_samples

            # Update the value function with the maximum Q-value for the agent
            best_action_value = np.max(Q[s])
            delta = max(delta, np.abs(V[s] - best_action_value))
            V[s] = best_action_value  # Update value function
        if iteration >= max_iters - 1: 
            print('Iterations filled')
        # Check for convergence
        if delta < epsilon:
            break

    # Extract best response policy
    best_response_policy = {(s, i): [1 / A] * A for s in range(S) for i in range(game.n)}

    for s in range(S):
        best_action = np.argmax(Q[s])  # Find the best action for the agent
        for i in range(game.n):
            if i == agent:
                best_response_policy[s, i] = [0] * A
                best_response_policy[s, i][best_action] = 1.0  # Set deterministic best response
            else:
                best_response_policy[s, i] = policies[s, i]  # Keep other agents' policies fixed

    return Q, best_response_policy
   

def compute_expected_value(game, policy, gamma=0.99, agent=0, kappa= "deterministic", epsilon=9e-3, max_iters=1000, num_samples=1000):
    """
    Computes the expected discounted value for a given agent under a specific policy.

    :param game: Game environment
    :param policy: Joint policy for all agents
    :param gamma: Discount factor
    :param agent: Index of the agent for whom we compute the expected value
    :param epsilon: Convergence threshold for value iteration
    :param max_iters: Maximum number of iterations for value iteration
    :param num_samples: Number of samples to estimate expectations over stochastic policies
    :return: The expected discounted value for the agent
    """
    S = game.S  # Number of states
    V = np.zeros(S)  # Initialize value function for the agent

    for iteration in range(max_iters):
        delta = 0
        # Loop over all states
        for s in range(S):
            expected_value = 0  # Initialize expected value for state 's'

            # Loop over all possible actions for the agent in state 's'
            for a in range(game.m):
                # Get the probability of the agent taking action 'a' in state 's'
                agent_action_probs = policy.get((s, agent), [1 / game.m] * game.m)
                action_prob = agent_action_probs[a]  # Get agent's probability for action 'a'

                # If the action probability is close to 0, skip it to save computation
                if action_prob > 0:
                    # Estimate the expected value for this action by sampling
                    sampled_value = 0
                    for _ in range(num_samples):
                        # Sample actions for other agents based on their policies
                        actions_others = [
                            np.random.choice(len(policy[(s, i)]), p=policy[(s, i)]) for i in range(game.n) if i != agent
                        ]

                        # Form the joint action including the current agent's action 'a'
                        actions_full = actions_others[:agent] + [a] + actions_others[agent:]

                        # Get the reward for the agent
                        reward = game.get_public_rewards(actions_full, s)[0]+game.xi(agent, s, a, game.n, game.epsilon)

                        # Sample the next state based on the joint action
                        next_state = game.sample_next_state(s, actions_full, kappa)

                        # Add discounted future value to the sampled value
                        sampled_value += reward + gamma * V[next_state]

                    # Average the sampled values over the number of samples
                    sampled_value /= num_samples

                    # Add the contribution of this action to the expected value for state 's'
                    expected_value += action_prob * sampled_value

            # Update the value function for state 's'
            delta = max(delta, np.abs(expected_value - V[s]))
            V[s] = expected_value

        # Check for convergence
        if delta < epsilon:
            break

    # Return the average expected value over all states for the agent
    return np.mean(V)

def compute_nash_regret(game, policies, gamma=0.9, kappa = "deterministic"):
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
        Q_opt, best_response_policy = compute_best_response(game, agent, policies, gamma=gamma, kappa=kappa)
        
        # Compute expected value of current policy and best response policy
        current_value = compute_expected_value(game, policies, gamma, agent, kappa=kappa)
        best_response_value = compute_expected_value(game, best_response_policy, gamma, agent, kappa=kappa)
        if best_response_value < current_value:
            print('Wait What')
            print("BR", best_response_value)
            print("curr", current_value)
        # Regret is the difference between best response and current policy value
        regret = best_response_value - current_value
        if regret >= total_regret:
            total_regret = regret
        print(agent, 'done')
    
    return total_regret



def full_experiment_PG(game, runs,iters,eta,T,samples, kappa, path, gamma):
    N ,  all_states, M, S= game.n, game.all_states, game.m, game.S
    # densities = np.zeros((S,M))
    raw_accuracies = []
    raw_regrets = []
    
    for k in tqdm.tqdm(range(runs)):
        policy_hist = policy_gradient(game, iters,gamma,eta,T,samples, kappa)
        raw_accuracies.append(get_accuracies(game, policy_hist))

        converged_policy = policy_hist[-1]
        # for i in range(N):
        #     for s in range(S):
        #         st = all_states[s]
        #         densities[s] += converged_policy[st,i]
        
        # Compute Nash regret for the final policy
        regret = compute_nash_regret(game, converged_policy, gamma, kappa)
        raw_regrets.append(regret)

    # Convert list of regrets to an array
    raw_regrets = np.array(raw_regrets)
    avg_regret = np.mean(raw_regrets)
    
    # Output the average Nash regret
    print("Average Nash regret:", avg_regret)
    print("Raw regret", raw_regrets)

    

    # # Plot Figure 1: trajectories of L1 accuracy
    # plot_accuracies = np.array(list(itertools.zip_longest(*raw_accuracies, fillvalue=np.nan))).T
    # clrs = sns.color_palette("husl", 3)
    # piters = list(range(plot_accuracies.shape[1]))
    # fig1 = plt.figure(figsize=(6,4))
    # for i in range(len(plot_accuracies)):
    #     plt.plot(piters, plot_accuracies[i])
    # plt.grid(linewidth=0.6)
    # plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N, runs,eta))
    # plt.show()
    # fig1.savefig(path + 'individual_runs_n{}.png'.format(N),bbox_inches='tight')


#     # Plot Figure 2: mean and std of L1 accuracy
#     plot_accuracies = np.nan_to_num(plot_accuracies)
#     if runs >= 2:
#         pmean = list(map(statistics.mean, zip(*plot_accuracies)))
#         pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))
#         fig2 = plt.figure(figsize=(6,4))
#         ax = sns.lineplot( pmean, color = clrs[0],label= 'Mean L1-accuracy')
#         ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[0],label="1-standard deviation")
#         ax.legend()
#         plt.grid(linewidth=0.6)
#         plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N, runs,eta))
#         plt.show()
#         fig2.savefig(path + 'avg_runs_n{}.png'.format(N),bbox_inches='tight')


# 	# Plot Figure 3: Density under different states
#     fig3 = game.density_plotting(densities)
#     plt.gca().set(xlabel='Facility',ylabel='Average number of agents', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(game.n,runs, eta))
#     plt.legend()
#     fig3.savefig(path+ 'facilities_n{}.png'.format(game.n),bbox_inches='tight')
#     plt.show()


    # # Save the matrix to a CSV file
    # np.savetxt(path + 'densities.csv', densities, delimiter=',')
    # np.savetxt(path +'plot_accu.csv', plot_accuracies, delimiter=',')


    return regret

