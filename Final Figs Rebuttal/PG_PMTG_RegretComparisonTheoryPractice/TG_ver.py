import numpy as np
import matplotlib.pyplot as plt
from helpers import logistic, bernoulli

class teamgame: 
    def __init__(self,n):
        self.n= n 
        self.epsilon = 0.1
        self.actions = [ np.random.choice(2, 1)[0] for i in range(n)]
        self.m = 2
        self.all_states = [0,1]
        self.S = len(self.all_states)
        self.act_dic = {0:0, 1:1}
        
    def get_counts(self,actions):
        return np.sum(actions)
        
    def get_public_rewards(self, actions, state):
        density = self.get_counts(actions)
        if density <= self.n/2:
            return 0,0
        else: 
            return 1,state
        
    def xi(self, i, s, a, n, epsilon):
        xi = (s == a) * ((n +1-i)/n)  * 10 - a * ((i+1)/n)
        return xi * epsilon
    
    def get_reward(self, actions, state, kappa):
        agents_rewards = self.n * [0]
        density, public_reward = self.get_public_rewards(actions,state)
        if density == 0:
            return agents_rewards
        for i in range(self.n):
            agents_rewards[i] = public_reward + self.xi(i, state, actions[i], self.n, self.epsilon)
        return agents_rewards

    def sample_next_state(self, state, actions, kappa="deterministic"):
        density = self.get_counts(actions)
        if kappa == "deterministic":
            if state == 0 and density >= self.n/2 or state == 1 and density >= self.n/4:
                return 1
            else: 
                return 0
        else:
            if state == 0:
                C = 1/2 + 1/(2*self.n)
            else:
                C = 1/4 + 1/(2*self.n)
            p = logistic(density, kappa, C, self.n)
            return bernoulli(p)