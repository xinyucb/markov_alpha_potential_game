import itertools as it
import numpy as np
from math import comb

from helpers import logistic, bernoulli

class EdgewiseCongGame:
	#inputs: num players, max facilities per player, list of linear multiplier on utility for num of players
	def __init__(self, n, d, weights):
		self.n = n # num of players
		self.d = d 
		self.weights = weights   # eg: [[1,-100],[2,-100],[4,-100],[6,-100]]   safe-state reward, unsafe-state penalty
		self.m = len(weights) #number of facilities
		self.num_actions = sum(comb(self.m,k) for k in range(1,self.d+1))
		self.facilities = [i for i in range(self.m)]
		self.actions = list(it.chain.from_iterable(it.combinations(self.facilities, r) for r in range(1,self.d+1)))
		#self.kappa = None

	def get_counts(self, actions):
		count = dict.fromkeys(range(self.m),0)
		for action in actions:
			for facility in action:
				count[facility] += 1
		return list(count.values())

	def get_facility_rewards(self, actions, states, kappa): # states = [0, 1, 1, 0] len(states) = num of facilities
		density = self.get_counts(actions)
		facility_rewards = self.m * [0]
		for j in range(self.m):
			curr_state_with_action =  next_state(states[j], density[j], threshold1=self.n/2, threshold2=self.n/4, kappa=kappa, N=self.n)
			facility_rewards[j] = density[j] * self.weights[j][0] + self.weights[j][curr_state_with_action]
		return facility_rewards

def get_agent_reward(game, actions, agent_action, states, kappa):
	agent_reward = 0
	facility_rewards = game.get_facility_rewards(actions, states, kappa)
	for facility in agent_action:
		agent_reward += facility_rewards[facility]
	return agent_reward

def get_reward(game, actions, states, kappa):
	rewards = game.n * [0]
	for i in range(game.n):
		rewards[i] = get_agent_reward(game, actions, actions[i], states, kappa)
	return rewards
    
def next_state(curr, density, threshold1, threshold2, kappa, N):
    # curr is 0 or 1
    # density is a number: the num of agents on a chosen facility
	if kappa == "deterministic":
		if curr == 0:
			if density > threshold1:
				return 1
			else:
				return 0
		if curr == 1:
			if density <= threshold2:
				return 0
			else:
				return 1
	else:
		if curr == 0:
			C = 1/2 + 1/(2*N)
		else:
			C = 1/4 + 1/(2*N)
		p = logistic(density, kappa, C, N)
		return bernoulli(p)