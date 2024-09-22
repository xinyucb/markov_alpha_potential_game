import itertools as it
import numpy as np
from itertools import product
from helpers import logistic, bernoulli
import matplotlib.pyplot as plt


class cong_game:
	#inputs: num players, max facilities per player, list of linear multiplier on utility for num of players
	def __init__(self, n, d, weights):
		self.n = n # num of players
		self.d = d 
		self.weights = weights   # eg: [[1,-100],[2,-100],[4,-100],[6,-100]]   safe-state reward, unsafe-state penalty
		self.m = len(weights) #number of facilities
		self.facilities = [i for i in range(self.m)]
		self.actions = list(it.chain.from_iterable(it.combinations(self.facilities, r) for r in range(1,self.d+1)))
		#self.kappa = None


		act_dic = {}
		counter = 0
		for act in self.actions:
			act_dic[counter] = act 
			counter += 1
		self.act_dic = act_dic

		self.all_states =  [i for i in product(range(2), repeat=self.m)]
		self.S = len(self.all_states)


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
			curr_state_with_action = self.next_state(states[j], density[j], threshold1=self.n/2, threshold2=self.n/4, kappa=kappa, N=self.n)
			facility_rewards[j] = density[j] * self.weights[j][0] + self.weights[j][curr_state_with_action]
		return facility_rewards

	def get_agent_reward(self, actions, agent_action, states, kappa):
		agent_reward = 0
		facility_rewards = self.get_facility_rewards(actions, states, kappa)
		for facility in agent_action:
			agent_reward += facility_rewards[facility]
		return agent_reward

	def get_reward(self, actions, states, kappa):
		rewards = self.n * [0]
		for i in range(self.n):
			rewards[i] = self.get_agent_reward(actions, actions[i], states, kappa)
		return rewards
		
	def next_state(self, curr, density, threshold1, threshold2, kappa, N):
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
		

	
	def sample_next_state(self, state, actions, kappa):
		# state: [0, 1, 0, 1]
		act_dic , N, D= self.act_dic, self.n, self.m
		acts_from_ints = [act_dic[i] for i in actions]
		density = self.get_counts(acts_from_ints)

		new_state = [0] * 4
		for j in range(D):
			new_state[j] = self.next_state(state[j], density[j], threshold1=N/2, threshold2=N/4, N=N, kappa=kappa)

		return tuple(new_state)
	

	def density_plotting(self, densities):
		fig3, ax = plt.subplots()
		index = np.arange(self.m)
		bar_width = 0.25
		opacity = 1

		id1 = self.all_states.index((0,0,0,0))
		rects1 = plt.bar(index, densities[id1], bar_width,
		alpha= .7 * opacity,
		color='b',
		label= str(self.all_states[id1]))

		id2 =  self.all_states.index((0,0,0,1))
		rects2 = plt.bar(index + bar_width, densities[id2], bar_width,
		alpha= opacity,
		color='y',
		label= str(self.all_states[id2]))

		id3 =  self.all_states.index((0,1,1,0))
		rects2 = plt.bar(index + bar_width + bar_width, densities[id3], bar_width,
		alpha= opacity,
		color='r',
		label= str(self.all_states[id3]))
		plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
		return fig3
