#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:59:18 2022

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt

# TODO: code environment
# Environment class
class Env:
    
    # initialization method
    def __init__(self, Q_env):
        self.Q_env = Q_env
        self.n_actions = len(Q_env)
    
    # generate environmental sample
    def sample(self, action):
        return self.Q_env[action]


# TODO: code agent
# // Initialize variables\;
#  $Q_0' \gets [0, 0, ..., 0]$\;
#  $Q_0^E \gets [0, 0, ..., 0]$\;
#  $Q_0^I \gets [0, 0, ..., 0]$\;
#  $t=0$\;
#  \While{termination condition not met}{
#   $T_t^E \gets$ singleSample($Q_t^R$)\; %, learning=frozen)\;
#   %policy=Boltzman($Q'$), $\alpha$=0)\;
#   $L_t'$ (performance) $\gets$ $L(\pi(Q_t',\tau'), T_t^E)$\; %reward($T_t^E$)\;
  
#   $L_t^E$ $\gets$ $L(\pi(Q_t^E, \tau^E), T_t^E)$\;
#   $L_t^I$ $\gets$ $L(\pi(Q_t^I, \tau^I), T_t^E)$\;
  
#   $\phi_t \gets L_t^I/(L_t^I + L_t^E)$\;
  
#   $Q_{t+1}^E \gets $ update($Q_t^E$ with $T_t^E, \pi, \tau^E, \alpha)$\;
#   $T_t^I \gets$ singleSample($Q_{t+1}^E$)\;%, learning=?)\;
#   $Q_{t+1}^I \gets $ update($Q_t^I$ with $T_t^I, \pi, \tau^I, \alpha)$\;
  
#   $Q_{t+1}' \gets \phi_t Q_{t+1}^E + (1-\phi_t) Q_{t+1}^I$\;
  
#   $t \gets t + 1$\;


# Learner class
class Learner:
    
    # initialization method
    def __init__(self, env, params):
        self.env = env
        self.n_actions = env.n_actions
        self.params = params
        self.Q_exp = np.zeros(self.n_actions)
        self.Q_img = np.zeros(self.n_actions)
        self.Q_mix = np.zeros(self.n_actions)
    
    def run(self, n_trials):
        rewards = []
        for trial in range(n_trials):
            reward = self.perform_trial()
            rewards.append(reward)
        print(self.Q_exp)
        print(self.Q_img)
        print(self.Q_mix)
        return rewards
    
    def perform_trial(self):
        # parameters
        # t_mix: tau used in mix Boltzmann policy
        # t_img: tau used in imagination Boltzmann policy
        # alpha: learning rate used in Q_exp and Q_img updates
        # phi: ovverrides adaptive Q_env vs Q_img weighting for computing Q_mix
        t_mix = self.params['t_mix']
        t_img = self.params['t_img']
        alpha = self.params['alpha']
        Q_exp = self.Q_exp
        Q_img = self.Q_img
        Q_mix = self.Q_mix
        
        # select action based on Q_mix (Boltzmann policy)
        act_probs = np.ones(self.n_actions) / self.n_actions
        for a in range(self.n_actions): # calculate Boltzmann action probs
            act_probs[a] = np.exp(t_mix * Q_mix[a]) / sum(np.exp(t_mix * Q_mix))
        action = np.random.choice(self.n_actions, p=act_probs) # choose action
        
        # evaluate performance
        reward = self.env.sample(action)
        
        # calculate loss wrt Q_exp
        loss_exp = abs(reward - Q_exp[action])
        
        # calculate loss wrt Q_img
        loss_img = abs(reward - Q_img[action])
        
        # calculate phi
        if 'phi' in self.params:
            phi = self.params['phi']
        elif loss_exp + loss_img == 0: # prevent divide by zero error
            phi = 0.5
        else:
            phi = loss_exp / (loss_exp + loss_img) # relative loss proportion
        
        # update Q_exp using exp sample
        self.Q_exp[action] += alpha * (reward - Q_exp[action])
        
        # sample using Q_exp
        img_act_probs = np.ones(self.n_actions) / self.n_actions
        for a in range(self.n_actions): # calculate Boltzmann action probs
            img_act_probs[a] = np.exp(t_img * Q_img[a]) / sum(np.exp(t_img * Q_img))
        img_action = np.random.choice(self.n_actions, p=img_act_probs) # choose action
        img_reward = self.Q_exp[img_action]
        
        # update Q_img using img sample
        self.Q_img[action] += alpha * (img_reward - Q_img[img_action])
        
        # update Q_mix using Q_exp, Q_img, and phi
        self.Q_mix = (1 - phi) * Q_exp + (phi) * Q_img
        
        # return reward for this trial
        return reward


# TODO: run simulations
e = Env([1,0,0,0])
params = {'t_mix': 0.0, 't_img': 1.0, 'alpha': 0.1, 'phi': 0.0}
rewards_record = []
for batch in range(100):
    agent = Learner(e, params)
    rewards = agent.run(100)
    rewards_record.append(rewards)
plt.plot(np.array(rewards_record).mean(axis=0))


# TODO: plot results

