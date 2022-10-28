#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:55:19 2022

@author: Jack
"""
## https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952

## There will specific file structuring if I want to "register" my gym env.

# import gym
# class BasicEnv(gym.Env):
# def __init__(self):
#         self.action_space = gym.spaces.Discrete(5)
#         self.observation_space = gym.spaces.Discrete(2)
# def step(self, action):
#         state = 1
    
#         if action == 2:
#             reward = 1
#         else:
#             reward = -1
            
#         done = True
#         info = {}
#         return state, reward, done, info
# def reset(self):
#         state = 0
#         return state

## Sure there us a way to implement gym environments. But are there any
## specifications for me to follow for structuring agents/algorithms?

## From Erik's code it seems I code as I like, as a usual class?

## And then code my own runs to use the two.

import numpy as np
import matplotlib.pyplot as plt
import math
import random 


## PROBLEM 2 : BANDITS
## In this section, we have given you a template for coding each of the 
## exploration algorithms: epsilon-greedy, optimistic initialization, UCB exploration, 
## and Boltzmann Exploration 

## You will be implementing these algorithms as described in the “10-armed Testbed” in Sutton+Barto Section 2.3
## Please refer to the textbook or office hours if you have any confusion.

## note: you are free to change the template as you like, do not think of this as a strict guideline
## as long as the algorithm is implemented correctly and the reward plots are correct, you will receive full credit

# This is the optional wrapper for exploration algorithm we have provided to get you started
# this returns the expected rewards after running an exploration algorithm in the K-Armed Bandits problem
# we have already specified a number of parameters specific to the 10-armed testbed for guidance
# iterations is the number of times you run your algorithm

# WRAPPER FUNCTION
def explorationAlgorithm(explorationAlgorithm, param, iters):
    cumulativeRewards = []
    for i in range(iters):
        # number of time steps
        t = 1000
        # number of arms, 10 in this instance
        k = 10
        # real reward distribution across K arms
        rewards = np.random.normal(1,1,k)
        # counts for each arm
        n = np.zeros(k)
        # extract expected rewards by running specified exploration algorithm with the parameters above
        # param is the different, specific parameter for each exploration algorithm
        # this would be epsilon for epsilon greedy, initial values for optimistic intialization, c for UCB, and temperature for Boltmann 
        currentRewards = explorationAlgorithm(param, t, k, rewards, n)
        cumulativeRewards.append(currentRewards)
    # TO DO: CALCULATE AVERAGE REWARDS ACROSS EACH ITERATION TO PRODUCE EXPECTED REWARDS
    expectedRewards = np.array(cumulativeRewards).mean(axis=0)
    return expectedRewards


# BOLTZMANN EXPLORATION TEMPLATE
def boltzmannE(temperature, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    currentRewards = np.zeros(steps)

    # TO DO: initialize q values per arm
    Q_table = np.zeros(k)

    # TO DO: initialize probability values for each arm
    action_probs = np.ones(k) / k

    # TO DO: implement the Boltzmann Exploration algorithm over all steps and return the expected rewards across all steps
    for step in range(steps):
        
        # calculate Boltzmann action probs based on Q table
        for a in range(k):
            action_probs[a] = np.exp(temperature * Q_table[a]) / np.exp(temperature * Q_table).sum()
        
        # log expected reward going into this time step
        currentRewards[step] = action_probs.dot(realRewards)
        
        # choose action
        action = np.random.choice(k, p=action_probs)
        
        # collect reward
        reward = realRewards[action] + np.random.normal()
        # track times action has been taken
        n[action] += 1
        # update action value estimate
        Q_table[action] += (1 / n[action]) * (reward - Q_table[action])
    
    #raise NotImplementedError()
    return currentRewards


# PLOT TEMPLATE
def plotExplorations(paramList, exploration_algo):
    # TO DO: for each parameter in the param list, plot the returns from the exploration Algorithm from each param on the same plot
    x = np.arange(1,1001)
    # calculate your Ys (expected rewards) per each parameter value
    # plot all the Ys on the same plot
    # include correct labels on your plot!
    
    plt.figure()
    
    for param in paramList:
        Y = explorationAlgorithm(exploration_algo, param, iters=100)
        plt.plot(x, Y, label=param)
    
    plt.xlabel('Steps')
    plt.ylabel('Average expected reward')

    if (exploration_algo == boltzmannE):               plt.legend(title='temperature')
    
    #raise NotImplementedError()


# RUN IT ALL
if __name__ == "__main__":
    plotExplorations([1, 3, 10, 30, 100], boltzmannE)
    
    plt.figure()
    
    
    
    
    
    
    
    
    