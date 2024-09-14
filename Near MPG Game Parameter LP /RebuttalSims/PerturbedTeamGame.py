import itertools
import numbers
import random
import typing
import numpy as np
from framework.game import *
from framework.utils import *

class perturbed_game:
    def __init__(self,  N: int,
        lambda_1: float = 0.8, lambda_2: float = 0.2, lambda_3: float = 0.8, lambda_4: float = 0.2, delta: float = 0.5,
        epsilon: float = 1) -> None:

        
        assert N >= 1

        self.LOW_EXCITEMENT = 0
        self.HIGH_EXCITEMENT = 1
        self.STATUSES = (self.LOW_EXCITEMENT, self.HIGH_EXCITEMENT)
        
        self.APPROVE = 1
        self.DISAPPROVE = 0
        self.APPROVAL_STATUS = (self.APPROVE, self.DISAPPROVE)

        self.PROJECT_HAPPEN = 1
        self.PROJECT_SHOVED = 0
    
        self.players = [Player(idx=i, label=str(i + 1)) for i in range(N)]
        self.I = PlayerSet(self.players)
        self.S = StateSet([State(value=v) for v in self.STATUSES])
        self.A = ActionProfileSet([ActionSet(i, [Action(i, value=j) for j in self.APPROVAL_STATUS]) for i in self.I])
        # 2 actions per player, either to approve or disapprove 
        self.N = N
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        
        self.delta = delta
        
        self.epsilon = epsilon
        
        transition_kernel = self.strategy_dependent_transition_kernel
        self.P = ProbabilityTransitionKernel(self.S, self.A, transition_kernel)


        self.mu = InitialStateDistribution(self.S, lambda s: 1 / len(self.S))
        self.R = RewardFunction(self.I, self.S, self.A, self.reward)
        self.game = StochasticGame(self.I, self.S, self.A, self.mu, self.P, self.R, delta)
        
    def reward(self, i, s, a):
        return self.get_reward(i, s, a)
    
    def xi(self, i, state, actions, N, epsilon):
    	"""
    	Free feel to modify this part
        """
    	xi = (actions[i].value == state) * ((N+1 -int(str(i)))/N)  * 5 - actions[i].value * ((int(str(i))+1)/N)
    	return xi * self.epsilon

    
    def get_reward(self, i, state, actions):
    	agents_rewards = 0
    
    	density, public_reward = self.get_public_rewards(state, actions)
    	if density == self.PROJECT_SHOVED:
    		return agents_rewards
    	
    	agents_rewards = public_reward + self.xi(i, state, actions, self.N, self.epsilon)
    	return agents_rewards
    
    def get_public_rewards(self, state, actions):
        density = int(sum(indicator(actions[j].value == self.APPROVE) for j in self.I))
        if density < self.N / 2:
            return self.PROJECT_SHOVED, 0 
        else:
            return self.PROJECT_HAPPEN, 1
    
    
    
    def strategy_dependent_transition_kernel(self, s, actions, s_prime):
        pr = 1.0
        
        num_approver = int(sum(indicator(actions[j].value == self.APPROVE) for j in self.I))
                
        if s == self.HIGH_EXCITEMENT: 
            if s_prime == self.HIGH_EXCITEMENT and num_approver >= self.N/4:
                pr *= self.lambda_1 
            if s_prime == self.LOW_EXCITEMENT and num_approver >= self.N/4:
                pr *= 1-self.lambda_1 
             
            if s_prime == self.HIGH_EXCITEMENT and num_approver < self.N/4:
                pr *= self.lambda_2
            if s_prime == self.LOW_EXCITEMENT and num_approver < self.N/4:
                pr *= 1-self.lambda_2
                
                
        if s == self.LOW_EXCITEMENT: 
            if s_prime == self.HIGH_EXCITEMENT and num_approver >= self.N/2:
                pr *= self.lambda_3 
            if s_prime == self.LOW_EXCITEMENT and num_approver >= self.N/2:
                pr *= 1-self.lambda_3 
             
            if s_prime == self.HIGH_EXCITEMENT and num_approver < self.N/2:
                pr *= self.lambda_4
            if s_prime == self.LOW_EXCITEMENT and num_approver < self.N/2:
                pr *= 1-self.lambda_4
            
        return pr

    

