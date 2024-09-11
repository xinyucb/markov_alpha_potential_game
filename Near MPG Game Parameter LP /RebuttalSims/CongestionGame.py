import itertools
import numbers
import random
import typing

from framework.game import *
from framework.utils import *

class congestion_game:

    def __init__(self,  N: int, M: int, U: int, m: typing.List[numbers.Number], 
                 b: typing.List[numbers.Number],
        lambda_1: float = 0.8, lambda_2: float = 0.2, delta: float = 0.5,
        common_interest: bool = False, strategy_independent_transitions: bool = False,
        seed: int = 0) -> None:

        if seed:
            random.seed(seed)
        else:
            random.seed(100)

        assert N >= 1
        assert M >= 1
        assert U >= 1

        self.SAFE_STATUS = 0
        self.UNSAFE_STATUS = 1
        self.STATUSES = (self.SAFE_STATUS, self.UNSAFE_STATUS)

        self.players = [Player(idx=i, label=str(i + 1)) for i in range(N)]
        self.I = PlayerSet(self.players)
        self.S = StateSet([State(value=a) for a in itertools.product(self.STATUSES, repeat=M)])
        self.A = ActionProfileSet([ActionSet(i, [Action(i, value=j) for j in range(M)]) for i in self.I])

        self.M = M
        self.N = N
        self.m = m
        self.b = b
        self.U = U
        self.common_interest = common_interest
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.delta = delta
        
        if strategy_independent_transitions:
            transition_matrix = {s: {s_prime: sum(self.strategy_dependent_transition_kernel(s, a, s_prime) 
                                                  for a in self.A) for s_prime in self.S} for s in self.S}
            for s in self.S:
                normalization = sum(transition_matrix[s][s_prime] for s_prime in self.S)
                for s_prime in self.S:
                    transition_matrix[s][s_prime] /= normalization

            def transition_kernel(s, a, s_prime):
                return transition_matrix[s][s_prime]

        else:
            transition_kernel = self.strategy_dependent_transition_kernel

        self.mu = InitialStateDistribution(self.S, lambda s: 1 / len(self.S))
        self.P = ProbabilityTransitionKernel(self.S, self.A, transition_kernel)
        self.R = RewardFunction(self.I, self.S, self.A, self.reward)
        self.game = StochasticGame(self.I, self.S, self.A, self.mu, self.P, self.R, delta)

    def reward_oneplayer(self, i, s, a):
        route = a[i].value
        status = s.value[route]
        multiplier = 1.0 if status == self.UNSAFE_STATUS else 2.0
        return self.b[route] - multiplier * self.m[route] * sum(indicator(a[j].value == route) for j in self.I)

    def reward(self, i, s, a):
        if self.common_interest:
            return sum(self.reward_oneplayer(i_prime, s, a) for i_prime in self.I)
        else:
            return self.reward_oneplayer(i, s, a)
        

    def reward_facility(self, s, a):
        """
        Calculate phi(s,a) = sum_{j in facilities} sum_{k=1}^{num of people at j with a} C(k; s, j)
        """
        phi = 0 
        for route in range(self.M):
            num_people_at_m = int(sum(indicator(a[j].value == route) for j in self.I))
            status = s.value[route]
            multiplier = 1.0 if status == self.UNSAFE_STATUS else 2.0
            for k in range(1, 1 + num_people_at_m):
                phi += self.b[route] - multiplier * self.m[route] * k
                
        return phi
    
    def strategy_dependent_transition_kernel(self, s, a, s_prime):
        pr = 1.0
        counts_a = {route: sum(indicator(a[i].value == route) for i in self.I) for route in range(self.M)}
        for route in range(self.M):
            a_status = self.UNSAFE_STATUS if counts_a[route] >= self.U else self.SAFE_STATUS
            s_prime_status = s_prime.value[route]
            if a_status == self.SAFE_STATUS and s_prime_status == self.SAFE_STATUS:
                pr *= self.lambda_1
            elif a_status == self.SAFE_STATUS and s_prime_status == self.UNSAFE_STATUS:
                pr *= 1 - self.lambda_1
            elif a_status == self.UNSAFE_STATUS and s_prime_status == self.SAFE_STATUS:
                pr *= self.lambda_2
            elif a_status == self.UNSAFE_STATUS and s_prime_status == self.UNSAFE_STATUS:
                pr *= 1 - self.lambda_2
        return pr

    

