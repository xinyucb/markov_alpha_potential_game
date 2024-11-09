#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:53:50 2023

@author: chinmaymaheshwari
"""

import cvxpy as cp 
from utils import *
from routing_game import *
from framework.game import *
from framework.utils import *
import copy
import math
import typing
import itertools
from tqdm import tqdm 

def LP_Formulation(N, M, U, m, b, lambda_1, lambda_2, delta,
               K, tau, alpha_r, beta_r, common_interest, strategy_independent):
    game = create_routing_game(N=N, M=M, U=U, m=m, b=b, lambda_1=lambda_1, lambda_2=lambda_2, delta=delta,
                                   common_interest=common_interest, strategy_independent_transitions=strategy_independent)
    
    I = game.I  # player set
    S = game.S  # state set
    A = game.A  # action profile set
    mu = game.mu  # initial state distribution
    P = game.P  # probability transition kernel
    R = game.R  # reward function
    delta = game.delta  # discount factor
    
    phi = {s: {a: cp.Variable() for a in A} for s in S}
    alpha= {s: cp.Variable() for s in S}
    y = cp.Variable(nonneg=True)
    
    constraints = [alpha[s] == 0 for s in S]
    objective = cp.Minimize(y)
    
    for i in I: 
        for s in S: 
            for a in A:
                for b in A:
                    mergedActionProfile = ActionProfile.merge(b[i], a.minus(i))
                    constraints.extend([phi[s][a]-phi[s][mergedActionProfile] 
                                        - R.get_reward(i, s, a) + R.get_reward(i, s, mergedActionProfile) <= alpha[s]])
                    constraints.extend([-phi[s][a]+phi[s][mergedActionProfile] 
                                        + R.get_reward(i, s, a) - R.get_reward(i, s, mergedActionProfile) <= alpha[s]])

    list_S = list(S)
    A_product = itertools.product(A, repeat=len(list_S))
    A_product_list = list()
    for a_product in A_product:
        a_product_dict = dict()
        for idx, s in enumerate(list_S):
            a_product_dict[s] = a_product[idx]
        A_product_list.append(a_product_dict)

    set_pure_policy = list()
    for a_product in A_product_list:
        pi = dict()
        for i in I:
            pi[i] = Policy(S, A[i], lambda s, ai: 1.0 if ai == a_product[s][i] else 0.0)
        pi = JointPolicy(pi)
        set_pure_policy.append(pi)
    
    num_pure_pol = len(set_pure_policy)
    
    countt = 1
    countt_v = 1
    for aa in tqdm(range(int(num_pure_pol))):
        for i in I:
            pi_i = set_pure_policy[aa][i]
            pi_minus_i = set_pure_policy[aa].minus(i)
            d_pi = construct_d_pi(i, pi_i, pi_minus_i, P, list_S, A, delta, mu)
            for bb in range(aa, int(num_pure_pol)):
                pi_b_i = set_pure_policy[bb][i]
                pi_b_minus_i = set_pure_policy[bb].minus(i)
                d_pi_b = construct_d_pi(i, pi_b_i, pi_minus_i, P, list_S, A, delta, mu)
                check_similarity = 0
                for j in I:
                    if j != i:
                        ker_i = pi_minus_i[j].kernel
                        ker_j = pi_b_minus_i[j].kernel
                        if ker_i == ker_j:
                            check_similarity += 1
                if check_similarity == len(I)-1:
                    # countt_v += 1
                    # print('number of times insider the if', countt_v)
                    # if d_pi.all() == d_pi_b.all(): 
                    #     print(countt)
                    #     countt += 1
                    constraints.extend([sum((d_pi_b[idxs,idxa]-d_pi[idxs,idxa])*(phi[s][a]-R.get_reward(i, s, a))
                                             for idxs, s in enumerate(S) for idxa, a in enumerate(A)) <= y])
                    constraints.extend([sum((d_pi_b[idxs,idxa]-d_pi[idxs,idxa])*(-phi[s][a]+R.get_reward(i, s, a)) 
                                            for idxs, s in enumerate(S) for idxa, a in enumerate(A)) <= y])
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)
    print(prob.value)
    
    return prob.value

def solve_LP():
    N = 3
    M = 2
    U = 1
    m = [2, 4]
    b = [9, 16]
    lambda_1 = 0.8
    lambda_2 = 0.2
    delta = 0.8
    K = int(1e5)
    tau = 1e-6
    alpha_r = 0.5
    beta_r = 1
    common_interest =  False
    strategy_independent = False
    a= LP_Formulation(N, M, U, m, b, lambda_1, lambda_2, delta, K, tau, alpha_r, beta_r,
               common_interest, strategy_independent)

solve_LP()