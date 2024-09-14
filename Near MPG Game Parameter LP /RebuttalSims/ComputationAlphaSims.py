#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:28:26 2024

@author: chinmaymaheshwari
"""

    
import matplotlib.pyplot as plt

#import cvxpy as cp 
from PerturbedTeamGame import perturbed_game
from utils import *
from routing_game import *
from framework.game import *
from framework.utils import *
import copy
import math
import typing
import itertools
from tqdm import tqdm 

def random_policy_single_player(i, S):
    dic = {}
    M = 2
    Ai = ActionSet(i, [Action(i, value=j) for j in range(M)]) 

    for s in S:
        # Generate a list of random numbers that sum to 1
        random_numbers = [random.uniform(0, 1) for _ in range(M)]
        total_sum = sum(random_numbers)

        # Normalize the numbers to ensure they sum to 1
        normalized_numbers = [num / total_sum for num in random_numbers]

        # Print the list of random numbers that sum to 1
        cnt = 0
        for ai in Ai:
            dic[s,ai] = normalized_numbers[cnt]
            cnt += 1
    return dic


def generate_random_pi(I, S, A):
    single_player_policies = {}
    for i in I:
        single_player_policies[i] = random_policy_single_player(i, S)

    def generate_single_player_random_policy(i, s, a):
        return single_player_policies[i][s,a]
    pi = dict()
    for i in I:
        pi[i] = Policy(S, A[i], lambda s, a: generate_single_player_random_policy(i, s, a))
    pi = JointPolicy(pi)
    return pi


def modify_one_players_policy(i, pi, pi_prime, I, S, A):
    new_pi = dict()
    for j in I:
        if j != i:
            new_pi[j] = Policy(S, A[j], lambda s, ai: pi[j][s, ai])
        else:
            new_pi[j] = Policy(S, A[j], lambda s, ai: pi_prime[j][s, ai])
    new_pi = JointPolicy(new_pi)
    return new_pi

def gi(i, pi, pi_prime, phi, R, P, S, list_S, A, delta, mu):
    pi_a_i = pi[i]
    pi_minus_i = pi.minus(i)
    d_pi_a = construct_d_pi(i, pi_a_i, pi_minus_i, P, list_S, A, delta, mu)

    pi_b_i = pi_prime[i]
    d_pi_b = construct_d_pi(i, pi_b_i, pi_minus_i, P, list_S, A, delta, mu)

    Num = 0
    for idxs, s in enumerate(S):
        for idxa, a in enumerate(A):
            Delta_d = d_pi_b[idxs,idxa]-d_pi_a[idxs,idxa]
            phi_s_a = phi[s][a]
            rew = R.get_reward(i, s, a)
            Num += Delta_d*(phi_s_a-rew)
    return Num


def g(pi, pi_prime,x, I, phi, R, P, S, list_S, A, delta, mu):
    max_val = 0
    for i in I:
        g_i = gi(i, pi, pi_prime, phi, R, P, S, list_S, A, delta, mu)
        if g_i > max_val:
            max_val = g_i
    return max_val -x

def find_best_potential_function_randomized(N, lambda_1, lambda_2, lambda_3, lambda_4, delta, alpha_r, beta_r, epsilon):
    game = perturbed_game(N=N, lambda_1=lambda_1, lambda_2=lambda_2, delta=delta,epsilon=epsilon)
 
    I = game.I  # player set
    S = game.S  # state set
    A = game.A  # action profile set
    mu = game.mu  # initial state distribution
    P = game.P  # probability transition kernel
    R = game.R  # reward function
    delta = game.delta  # discount factor
    list_S = list(S)

    phi = {s: {a: 0 for a in A} for s in S}
    
    for s in S: 
        for a in A: 
            den, phi[s][a] = game.get_public_rewards(s, a)

    def gamma_(n):
        return  1/(5* n) 

    def delta_(n):
        return (n)**0.45 



    T = 100000
    X = 1
    X_list = [X]
    for n in range(T+1):
        pi = generate_random_pi(I, S, A)
        pi_prime = generate_random_pi(I, S, A)
        g_val = g(pi, pi_prime, X, I, phi, R, P, S, list_S, A, delta, mu)
        X_new = X - gamma_(n+1) * (1 - delta_(n+1) * (abs(g_val) > X) * (abs(g_val) - X))

        X_list.append(X_new)
        X = X_new
        if n%100 == 0:
            print(n, X)

    return X_list


N = 3

lambda_1 = 0.8
lambda_2 = 0.2

lambda_3 = 0.8
lambda_4 = 0.2


delta = 0.99

epsilon = 0.1

alpha_r = 0.4
beta_r = 0.8

X_list = find_best_potential_function_randomized(N, lambda_1, lambda_2, lambda_3, lambda_4, delta, alpha_r, beta_r, epsilon)
plt.plot(X_list)
plt.xlabel(r"$t$")
plt.ylabel(r"$y_t$")
plt.title("Perturbed Team Game: agents = %s"%str(N))
plt.show()