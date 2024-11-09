import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns; sns.set()
import statistics
import numpy as np
import tqdm
import numpy as np
import copy



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

def value_function(game, policy, gamma, T,samples, kappa,tau):
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
                # setdefault(key, value): if key exists in dic, return its original value in dic. Else, add this new key and value into dic 
                rewards = selected_profiles.setdefault(q, game.get_reward([act_dic[i] for i in actions], curr_state, kappa))                  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i] - tau*(gamma**t)*entropy(policy,curr_state,i)
                curr_state = game.sample_next_state(curr_state, actions, kappa)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(game, agent, state, action, policy, gamma, value_fun, samples, kappa,tau,actset):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    act_dic , N = game.act_dic, game.n
    selected_profiles = {}
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        if actset ==1:
            actions[agent] = action
            
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q, game.get_reward( [act_dic[i] for i in actions], state, kappa))                  
        tot_reward += rewards[agent] - tau*entropy(policy,state,agent) + gamma*value_fun[game.sample_next_state(state, actions, kappa), agent]
    return (tot_reward / samples)


def Q_function_one_step_policy(game, agent, state, policy, policy_one_step , gamma, value_fun, samples, kappa,tau):
    """
    Q = r(s, ai) + gamma * V(s)
    """
    act_dic , N = game.act_dic, game.n
    selected_profiles = {}
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q, game.get_reward([act_dic[i] for i in actions], state, kappa))                  
        tot_reward += rewards[agent] - tau*entropy(policy,state,agent)- tau*entropy(policy_one_step,state,agent) + gamma*value_fun[game.sample_next_state( state, actions, kappa), agent]
    return (tot_reward / samples)

def policy_accuracy(game, policy_pi, policy_star):
    N ,  all_states= game.n, game.all_states
    total_dif = N * [0]
    total_dif = N * [0]
    for agent in range(N):
        for state in all_states:
            total_dif[agent] += np.sum(np.abs((np.array(policy_pi[state, agent]) - np.array(policy_star[state, agent]))))
    return np.sum(total_dif) / N

def sequential_br(game, max_iters, gamma, T, samples,kappa,tau_, tau_discounting):
    N ,  all_states, M, S, all_states =  game.n, game.all_states, game.m, game.S, game.all_states

    policy = {(s,i): [1/M]*M for s in all_states for i in range(N)}
    policy_exp = {(s,i): [1/M]*M for s in all_states for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in tqdm.tqdm(range(max_iters)):
        tau = tau_ *  tau_discounting**(t+1)
        b_dist = M * [1]
            
        grads_exp = np.zeros((N, S, M))
        grads_diff = np.zeros((N,S))
        grads = np.zeros((N, S, M))
        value_fun = value_function(game, policy, gamma, T, samples, kappa,tau)
	
        # Computing the possible improvement for all players 
        for agent in range(N):
            for s in range(S):
                for act in range(M):
                    st = all_states[s]
                    grads_exp[agent, s, act] =  Q_function(game, agent, st, act, policy, gamma, value_fun, samples, kappa, tau,1)
        
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
                fac1 = Q_function_one_step_policy(game, agent, st, policy, policy_exp, gamma, value_fun, samples, kappa, tau)
                fac2 = Q_function_one_step_policy(game, agent, st, policy, policy, gamma, value_fun, samples, kappa, tau)
                grads_diff[agent, s] =  fac1-fac2
                                            
        max_index = np.where(grads_diff==np.max(grads_diff))
        agent_max = max_index[0][0]
        s_max = max_index[1][0]

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


    return policy_hist



def get_accuracies(game, policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(game, policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies


def full_experiment_BR(game, runs,iters, T,samples, kappa, tau, tau_discounting, path):
    N ,  all_states, M, S= game.n, game.all_states, game.m, game.S, 
    densities = np.zeros((S,M))

    raw_accuracies = [] 
    for k in tqdm.tqdm(range(runs)):
        policy_hist = sequential_br(game, iters,0.99, T,samples, kappa,tau, tau_discounting)
        raw_accuracies.append(get_accuracies(game, policy_hist))

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

    fig1 = plt.figure(figsize=(6,4))
    for i in range(len(plot_accuracies)):
        plt.plot(piters, plot_accuracies[i])
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Seq BR: agents = {}, runs = {}, '.format(N, runs) + r'$\tau$ = '+str(tau))
    plt.show()
    fig1.savefig(path + 'individual_runs_n{}.png'.format(N),bbox_inches='tight')
    plt.close()

    plot_accuracies = np.nan_to_num(plot_accuracies)
    if runs >= 2:
        pmean = list(map(statistics.mean, zip(*plot_accuracies)))
        pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))
        
        fig2 = plt.figure(figsize=(6,4))
        ax = sns.lineplot( pmean, color = clrs[0],label= 'Mean L1-accuracy')
        ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[0],label="1-standard deviation")
        ax.legend()
        plt.grid(linewidth=0.6)
        plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Sequential BR: agents = {}, runs = {}, '.format(N, runs) + r'$\tau$ = '+str(tau))
        plt.show()
        fig2.savefig(path+'avg_runs_n{}.png'.format(N),bbox_inches='tight')
        plt.close()

    # Plot Figure 3: Density under different states
    fig3 = game.density_plotting(densities)
    plt.gca().set(xlabel='Facility',ylabel='Average number of agents', title='Sequential Best Response: agents = {}, runs = {}, '.format(N, runs) + r'$\tau$ = '+str(tau))
    plt.legend()
    fig3.savefig(path+ 'facilities_n{}.png'.format(N),bbox_inches='tight')
    plt.show()
  
    # Save the matrix to a CSV file
    np.savetxt(path + 'densities.csv', densities, delimiter=',')
    np.savetxt(path +'plot_accu.csv', plot_accuracies, delimiter=',')


    return densities, plot_accuracies
    