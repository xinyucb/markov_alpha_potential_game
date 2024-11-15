import numpy as np
from math import comb
import tqdm
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
        self.epsilon = 10
        self.actions = [np.random.choice(2, 1)[0] for i in range(n)]
        self.num_actions = 2  # Number of actions is the same for all agents
        self.all_states = [0, 1]
        self.S = len(self.all_states)  # Number of states

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
    xi_val = (s == a) * ((n + 1 - i) / n) * 5 - a * ((i + 1) / n)
    return xi_val * epsilon

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
       Low:  density < n/2
"""
def sample_next_state(game, state, actions):
    """Deterministic transition"""
    density = game.get_counts(actions)
    if (state == 0 and density >= game.n / 2) or (state == 1 and density >= game.n / 4):
        return 1
    return 0

def pick_action(prob_dist):
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p=prob_dist)
    return action[0]

def entropy(policy, curr_state, num_player):
    ent = 0
    for player_index in range(num_player):
        for a in range(len(policy[curr_state, player_index])):
            if policy[curr_state, player_index][a] >= 1e-6:
                ent += policy[curr_state, player_index][a] * np.log(policy[curr_state, player_index][a])
    return ent

def value_function(game, policy, gamma, T, samples, tau):
    """
    O(num_samples * S * T) 
    Get value function by generating trajectories and calculating the rewards
    Vi(s) = sum_{t<T} gamma^t r(t)
    """
    S = game.S
    N = game.n
    
    selected_profiles = {}
    value_fun = {(s, i): 0 for s in range(S) for i in range(N)}
    for k in range(samples):
        for state in range(S):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]  # N: num of players
                q = tuple(actions + [curr_state])
                rewards = selected_profiles.setdefault(q, get_reward(game, actions, state))                  
                for i in range(N):
                    value_fun[state, i] += (gamma**t) * rewards[i] - tau * (gamma**t) * entropy(policy, curr_state, N)
                curr_state = sample_next_state(game, curr_state, actions)
    value_fun.update((x, v / samples) for (x, v) in value_fun.items())
    return value_fun

def Q_function(game, agent, state, action, policy, gamma, value_fun, samples, tau, actset):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    selected_profiles = {}
    N = game.n
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        if actset == 1:
            actions[agent] = action
        q = tuple(actions + [state])
        rewards = selected_profiles.setdefault(q, get_reward(game, actions, state))
        tot_reward += rewards[agent] - tau * entropy(policy, state, N) + gamma * value_fun[sample_next_state(game, state, actions), agent]
    return tot_reward / samples

def Q_function_one_step_policy(game, agent, state, policy, policy_one_step, gamma, value_fun, samples, tau):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    selected_profiles = {}
    N = game.n
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        q = tuple(actions + [state])
        rewards = selected_profiles.setdefault(q, get_reward(game, actions, state))
        tot_reward += rewards[agent] - tau * entropy(policy, state, N) - tau * entropy(policy_one_step, state, agent) + gamma * value_fun[sample_next_state(game, state, actions), agent]
    return tot_reward / samples

def policy_accuracy(policy_pi, policy_star, N, S):
    total_dif = N * [0]
    for agent in range(N):
        for state in range(S):
            total_dif[agent] += np.sum(np.abs((np.array(policy_pi[state, agent]) - np.array(policy_star[state, agent]))))
    return np.sum(total_dif) / N

def sequential_br(game, max_iters, gamma, eta, T, samples, kappa, M, S, N, tau):
    policy = {(s, i): [1/M] * M for s in range(S) for i in range(N)}
    policy_exp = {(s, i): [1/M] * M for s in range(S) for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]
    
    for t in tqdm.tqdm(range(max_iters)):
        grads_exp = np.zeros((N, S, M))
        grads_diff = np.zeros((N, S))
        value_fun = value_function(game, policy, gamma, T, samples, tau)

        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads_exp[agent, st, act] = Q_function(game, agent, st, act, policy, gamma, value_fun, samples, tau, 1)
        
        for agent in range(N):
            for st in range(S):
                sum_s_agent = 0
                max_val = np.max(grads_exp[agent, st])
                for a in range(M):
                    sum_s_agent += np.exp((grads_exp[agent, st, a] - max_val) / tau)
                for a in range(M):
                    policy_exp[st, agent][a] = np.exp((grads_exp[agent, st, a] - max_val) / tau) / sum_s_agent

        for agent in range(N):
            for st in range(S):
                fac1 = Q_function_one_step_policy(game, agent, st, policy, policy_exp, gamma, value_fun, samples, tau)
                fac2 = Q_function_one_step_policy(game, agent, st, policy, policy, gamma, value_fun, samples, tau)
                grads_diff[agent, st] = fac1 - fac2

        max_index = np.where(grads_diff == np.max(grads_diff))
        agent_max = max_index[0][0]
        s_max = max_index[1][0]
        
        for agent in range(N):
            for st in range(S):
                if agent == agent_max and st == s_max and grads_diff[agent, st] >= 0:
                    max_val = np.max(grads_exp[agent, st])
                    sum_s_agent = 0
                    for a in range(M):
                        sum_s_agent += np.exp((grads_exp[agent, st, a] - max_val) / tau)
                    for a in range(M):
                        policy[st, agent][a] = np.exp((grads_exp[agent, st, a] - max_val) / tau) / sum_s_agent

        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(policy_hist[t], policy_hist[t-1], N, S) < 10e-16:
            return policy_hist

    return policy_hist

def get_accuracies(policy_hist, N, S):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin, N, S)
        accuracies.append(this_acc)
    return accuracies

def compute_best_response(game, agent, policies, gamma=0.99, epsilon=9e-3, max_iters=1000, num_samples=1000):
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
    A = game.num_actions  # Number of actions

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
                    reward = game.get_public_rewards(actions_full, s)[0] + xi(agent, s, a, game.n, game.epsilon)

                    # Sample the next state based on the joint action
                    next_state = sample_next_state(game, s, actions_full)

                    # Add discounted future value to the expected value
                    expected_value += reward + gamma * V[next_state]

                # Average the sampled values
                Q[s, a] = expected_value / num_samples

            # Update the value function with the maximum Q-value for the agent
            best_action_value = np.max(Q[s])
            delta = max(delta, np.abs(V[s] - best_action_value))
            V[s] = best_action_value  # Update value function
        print(iteration, delta)
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
   

def compute_expected_value(game, policy, gamma=0.99, agent=0, epsilon=9e-3, max_iters=1000, num_samples=1000):
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
            for a in range(game.num_actions):
                # Get the probability of the agent taking action 'a' in state 's'
                agent_action_probs = policy.get((s, agent), [1 / game.num_actions] * game.num_actions)
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
                        reward = game.get_public_rewards(actions_full, s)[0]+xi(agent, s, a, game.n, game.epsilon)

                        # Sample the next state based on the joint action
                        next_state = sample_next_state(game, s, actions_full)

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

def compute_nash_regret(game, policies, gamma=0.9):
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
        current_value = compute_expected_value(game, policies, gamma, agent)
        best_response_value = compute_expected_value(game, best_response_policy, gamma, agent)
        if best_response_value <= current_value:
            print('Wait What')
        # Regret is the difference between best response and current policy value
        regret = best_response_value - current_value
        if regret >= total_regret:
            total_regret = regret
        print(agent, 'done')
    
    return total_regret

def full_experiment_with_regret(game, runs, iters, eta, T, samples, kappa, tau, gamma):
    """
    Runs the full experiment, computing policies, accuracies, and Nash regrets over multiple runs.
    """
    S, N, all_states, M = game.S, game.n, game.all_states, game.num_actions
    path = "team_model_results/kappa_" + str(kappa) + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    raw_accuracies = []
    raw_regrets = []
    
    
    for k in tqdm.tqdm(range(runs)):
        policy_hist = sequential_br(game, iters, gamma, eta, T, samples, kappa, M, S, N, tau)
        raw_accuracies.append(get_accuracies(policy_hist, N, S))

        # Debugging: Check the structure of the final policy
        final_policy = policy_hist[-1]
        
        # Compute Nash regret for the final policy
        regret = compute_nash_regret(game, final_policy, gamma)
        raw_regrets.append(regret)

    # Convert list of regrets to an array
    raw_regrets = np.array(raw_regrets)
    avg_regret = np.mean(raw_regrets)
    
    # Output the average Nash regret
    print("Average Nash regret:", avg_regret)
    print("Raw regret", raw_regrets)
    return avg_regret



# Run the experiment
game = TeamGame(3)
game.epsilon = 0.1
runs, iters, eta, T, samples, kappa, tau, gamma = 1, 100, 0.01, 100, 10, "deterministic", 1, 0.99

full_experiment_with_regret(game, runs, iters, eta, T, samples, kappa, tau, gamma)
