#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:59:18 2022

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt

from bandit import DeceptiveBanditOneHigh10
from bandit import BanditHardAndSparse2
from bandit import BanditHardAndSparse10
from bandit import BanditHardAndSparse121
from bandit import BanditHardAndSparse1000


# Environment class
class Env:
    
    # initialization method
    def __init__(self, Q_env):
        self.Q_env = Q_env
        #self.n_actions = len(Q_env)
        
        ActionSpace = type('MyObject', (object,), {})
        self.action_space = ActionSpace()
        self.action_space.n = len(Q_env)
        
        self.counter = 0
        self.changepoint = 50
    
    # generate environmental sample
    def step(self, action):
        if self.counter == self.changepoint: self.Q_env.reverse()
        self.counter += 1
        observation = None
        reward = self.Q_env[action]
        done = None
        info = None
        return (observation, reward, done, info)

class RandomWalkEnv:
    
    # initialization method
    def __init__(self, Q_envs):
        self.Q_envs = Q_envs
        #self.n_actions = len(Q_env)
        
        ActionSpace = type('MyObject', (object,), {})
        self.action_space = ActionSpace()
        self.action_space.n = len(Q_envs[0])
        
        self.counter = 0
    
    # generate environmental sample
    def step(self, action):
        observation = None
        reward = self.Q_envs[self.counter][action]
        done = None
        info = None
        self.counter += 1
        return (observation, reward, done, info)

# Learner class
class Learner:
    
    # initialization method
    def __init__(self, env, params):
        self.env = env
        self.n_actions = env.action_space.n #env.n_actions
        self.params = params
        self.Q_exp = np.zeros(self.n_actions)
        self.Q_img = np.zeros(self.n_actions)
        self.Q_mix = np.zeros(self.n_actions)
        self.env_steps_taken = 0
    
    def run(self, n_trials):
        rewards = []
        phis = []
        for trial in range(n_trials):
            (reward, phi) = self.perform_trial()
            rewards.append(reward)
            phis.append(phi)
        # print(self.Q_exp)
        # print(self.Q_img)
        # print(self.Q_mix)
        return (rewards, phis)
    
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
        (_, reward, _, _) = self.env.step(action)
        self.env_steps_taken += 1
        
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
        
        if self.env_steps_taken == 1: phi = 0 # phi is 0 for the 1st time step
        
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
        return (reward, phi)




# # params = {'t_mix': 3.0, 't_img': 3.0, 'alpha': 0.1, 'phi': 0.0}
# # rewards_record = []
# # for batch in range(1000):
# #     e = Env([1,0,0,0])
# #     agent = Learner(e, params)
# #     rewards = agent.run(100)
# #     rewards_record.append(rewards)
# # plt.plot(np.array(rewards_record).mean(axis=0))


# # generate random walk reward schedules
# A_init = np.random.rand()
# Qs = [np.array([A_init, 0.2, 0.3, 1 - A_init])]
# #start = np.random.rand(4)
# for i in range(99):
#     drift = np.random.normal(0, 0.1) #, 4)
#     Qs.append(Qs[-1] + [drift, 0, 0, -1 * drift])
#     if Qs[-1][0] > 1:
#         Qs[-1][0] = 1
#         Qs[-1][3] = 0
#     if Qs[-1][0] < 0:
#         Qs[-1][0] = 0
#         Qs[-1][3] = 1


# fig, axs = plt.subplots(3, sharex=True, figsize=(8,10))

# axs[0].plot(np.array(Qs)[:,0], c='C6', lw=4, alpha=0.6, label='A')
# axs[0].plot(np.array(Qs)[:,1], c='C5', lw=4, alpha=0.6, label='B')
# axs[0].plot(np.array(Qs)[:,2], c='C8', lw=4, alpha=0.6, label='C')
# axs[0].plot(np.array(Qs)[:,3], c='C9', lw=4, alpha=0.6, label='D')


# # axs[0].plot(np.concatenate([np.full(50,1),np.full(50,0)]), c='C6', lw=4, alpha=0.6, label='A')
# # axs[0].plot(np.full(100,0),                                c='C5', lw=4, alpha=0.6, label='B')
# # axs[0].plot(np.full(100,0),                                c='C8', lw=4, alpha=0.6, label='C')
# # axs[0].plot(np.concatenate([np.full(50,0),np.full(50,1)]), c='C9', lw=4, alpha=0.6, label='D')

# axs[0].set_title('4-armed bandit task structure')
# axs[0].set_ylabel('Reward') #'P(reward)')
# axs[0].legend(title='arm', loc='upper left')

# rewards_by_agent = []

# for phi in [0.0, 0.25, 0.5, 0.75, 1.0]:
#     params = {'t_mix': 3.0, 't_img': 3.0, 'alpha': 0.1, 'phi': phi}
#     rewards_record = []
#     phis_record = []
#     for batch in range(1000):
#         e = RandomWalkEnv(Qs) #BanditHardAndSparse2() #DeceptiveBanditOneHigh10() #Env([1,0,0,0])
#         agent = Learner(e, params)
#         (rewards, phis) = agent.run(100)
#         rewards_record.append(rewards)
#         phis_record.append(phis)
#     rewards_by_agent.append(rewards_record)
#     means = np.array(rewards_record).mean(axis=0)
#     stds = np.array(rewards_record).std(axis=0)
#     axs[1].plot(means, label=str(phi))
#     axs[1].fill_between(np.arange(100), means+stds, means-stds, alpha=0.2)

# params = {'t_mix': 3.0, 't_img': 3.0, 'alpha': 0.1}
# rewards_record = []
# phis_record = []
# for batch in range(1000):
#     e = RandomWalkEnv(Qs) #BanditHardAndSparse2() #DeceptiveBanditOneHigh10() #Env([1,0,0,0])
#     agent = Learner(e, params)
#     (rewards, phis) = agent.run(100)
#     rewards_record.append(rewards)
#     phis_record.append(phis)
# rewards_by_agent.append(rewards_record)
# means = np.array(rewards_record).mean(axis=0)
# stds = np.array(rewards_record).std(axis=0)
# axs[1].plot(means, label='auto', c='k', linestyle='dashed')
# axs[1].fill_between(np.arange(100), means+stds, means-stds, color='k', alpha=0.1)

# #fig.suptitle('Avg. reward over 1,000 runs +- std')
# axs[1].set_title('Avg. reward over 1,000 runs +- std')
# axs[1].set_ylabel('reward')
# axs[1].legend(title='phi setting', loc='upper left')

# means = np.array(phis_record).mean(axis=0)
# stds = np.array(phis_record).std(axis=0)
# axs[2].plot(means, label='phi', c='k')
# axs[2].fill_between(np.arange(100), means+stds, means-stds, color='k', alpha=0.1)
# axs[2].set_title('Avg. adaptive phi setting over 1,000 runs +- std')
# axs[2].set_ylim(0,1)
# axs[2].set_ylabel('phi')
# axs[2].set_xlabel('trial')
# axs[2].legend(loc='upper left')
 
# plt.show()


# # fig, axs = plt.subplots(3)
# # for phi in [0.0, 0.25, 0.5, 0.75, 1.0]:

# #     #fig.suptitle('Vertically stacked subplots')
# #     axs[0].plot(x, y)
# #     axs[1].plot(x, -y)

# fig, ax = plt.subplots(1, figsize=(8,4))

# batch_means = []
# for agent_data in rewards_by_agent:
#     avg_rewards = []
#     for batch_rewards in agent_data:
#         avg_rewards.append(np.array(batch_rewards).mean())
#     batch_means.append(np.array(avg_rewards).mean())

# ax.bar(np.arange(6), batch_means)
# ax.set_xticklabels(labels=['0.0', '0.0', '0.25', '0.5', '0.75', '1.0', 'auto'])
# ax.set_xlabel('Phi setting')
# ax.set_ylabel('Average experiment-averaged reward')

# plt.show()









# # # generate sparse random reward schedules
# # arm_probs = [0.1, 0.2, 0.1, 0.2]
# # Qs = []
# # for i in range(100):
# #     Qs.append((np.random.rand(4) < arm_probs).astype(int))


# fig, axs = plt.subplots(3, sharex=True, figsize=(8,10))

# # axs[0].plot(np.array(Qs)[:,0], c='C6', lw=4, alpha=0.6, label='A')
# # axs[0].plot(np.array(Qs)[:,1], c='C5', lw=4, alpha=0.6, label='B')
# # axs[0].plot(np.array(Qs)[:,2], c='C8', lw=4, alpha=0.6, label='C')
# # axs[0].plot(np.array(Qs)[:,3], c='C9', lw=4, alpha=0.6, label='D')


# axs[0].plot(np.full(100,0.1), c='C6', lw=4, alpha=0.6, label='A')
# axs[0].plot(np.full(100,0.2), c='C5', lw=4, alpha=0.6, label='B')
# axs[0].plot(np.full(100,0.1), c='C8', lw=4, alpha=0.6, label='C')
# axs[0].plot(np.full(100,0.1), c='C9', lw=4, alpha=0.6, label='D')

# axs[0].set_title('4-armed bandit task structure')
# axs[0].set_ylabel('Reward') #'P(reward)')
# axs[0].legend(title='arm', loc='upper left')
# axs[0].set_ylim(0,1)

# rewards_by_agent = []

# for phi in [0.0, 0.25, 0.5, 0.75, 1.0]:
#     params = {'t_mix': 3.0, 't_img': 3.0, 'alpha': 0.1, 'phi': phi}
#     rewards_record = []
#     phis_record = []
#     for batch in range(1000):
#         arm_probs = [0.1, 0.2, 0.1, 0.2]
#         Qs = []
#         for i in range(100):
#             Qs.append((np.random.rand(4) < arm_probs).astype(int))
#         e = RandomWalkEnv(Qs) #BanditHardAndSparse2() #DeceptiveBanditOneHigh10() #Env([1,0,0,0])
#         agent = Learner(e, params)
#         (rewards, phis) = agent.run(100)
#         rewards_record.append(rewards)
#         phis_record.append(phis)
#     rewards_by_agent.append(rewards_record)
#     means = np.array(rewards_record).mean(axis=0)
#     stds = np.array(rewards_record).std(axis=0)
#     axs[1].plot(means, label=str(phi))
#     axs[1].fill_between(np.arange(100), means+stds, means-stds, alpha=0.2)

# params = {'t_mix': 3.0, 't_img': 3.0, 'alpha': 0.1}
# rewards_record = []
# phis_record = []
# for batch in range(1000):
#     arm_probs = [0.1, 0.2, 0.1, 0.2]
#     Qs = []
#     for i in range(100):
#         Qs.append((np.random.rand(4) < arm_probs).astype(int))
#     e = RandomWalkEnv(Qs) #BanditHardAndSparse2() #DeceptiveBanditOneHigh10() #Env([1,0,0,0])
#     agent = Learner(e, params)
#     (rewards, phis) = agent.run(100)
#     rewards_record.append(rewards)
#     phis_record.append(phis)
# rewards_by_agent.append(rewards_record)
# means = np.array(rewards_record).mean(axis=0)
# stds = np.array(rewards_record).std(axis=0)
# axs[1].plot(means, label='auto', c='k', linestyle='dashed')
# axs[1].fill_between(np.arange(100), means+stds, means-stds, color='k', alpha=0.1)

# #fig.suptitle('Avg. reward over 1,000 runs +- std')
# axs[1].set_title('Avg. reward over 1,000 runs +- std')
# axs[1].set_ylabel('Experienced reward')
# axs[1].legend(title='phi setting', loc='upper left')

# means = np.array(phis_record).mean(axis=0)
# stds = np.array(phis_record).std(axis=0)
# axs[2].plot(means, label='phi', c='k')
# axs[2].fill_between(np.arange(100), means+stds, means-stds, color='k', alpha=0.1)
# axs[2].set_title('Avg. adaptive phi setting over 1,000 runs +- std')
# axs[2].set_ylim(0,1)
# axs[2].set_ylabel('phi')
# axs[2].set_xlabel('trial')
# axs[2].legend(loc='upper left')
 
# plt.show()



# fig, ax = plt.subplots(1, figsize=(8,4))

# batch_means = []
# for agent_data in rewards_by_agent:
#     avg_rewards = []
#     for batch_rewards in agent_data:
#         avg_rewards.append(np.array(batch_rewards).mean())
#     batch_means.append(np.array(avg_rewards).mean())

# ax.bar(np.arange(6), batch_means)
# ax.set_xticklabels(labels=['0.0', '0.0', '0.25', '0.5', '0.75', '1.0', 'auto'])
# ax.set_xlabel('Phi setting')
# ax.set_ylabel('Average experiment-averaged reward')

# plt.show()











# generate deceptive reward schedules

Qs = []
for i in range(20):
    Qs.append(np.array([.2,.9,.2,.2]))
for i in range(10):
    Qs.append(np.array([.2,.1,.2,.2]))
for i in range(70):
    Qs.append(np.array([.2,.9,.2,.2]))


fig, axs = plt.subplots(3, sharex=True, figsize=(8,10))

axs[0].plot(np.array(Qs)[:,0], c='C6', lw=4, alpha=0.6, label='A')
axs[0].plot(np.array(Qs)[:,1], c='C5', lw=4, alpha=0.6, label='B')
axs[0].plot(np.array(Qs)[:,2], c='C8', lw=4, alpha=0.6, label='C')
axs[0].plot(np.array(Qs)[:,3], c='C9', lw=4, alpha=0.6, label='D')


# axs[0].plot(np.concatenate([np.full(50,1),np.full(50,0)]), c='C6', lw=4, alpha=0.6, label='A')
# axs[0].plot(np.full(100,0),                                c='C5', lw=4, alpha=0.6, label='B')
# axs[0].plot(np.full(100,0),                                c='C8', lw=4, alpha=0.6, label='C')
# axs[0].plot(np.concatenate([np.full(50,0),np.full(50,1)]), c='C9', lw=4, alpha=0.6, label='D')

axs[0].set_title('4-armed bandit task structure')
axs[0].set_ylabel('Reward') #'P(reward)')
axs[0].legend(title='arm', loc='upper left')

rewards_by_agent = []

for phi in [0.0, 0.25, 0.5, 0.75, 1.0]:
    params = {'t_mix': 3.0, 't_img': 3.0, 'alpha': 0.1, 'phi': phi}
    rewards_record = []
    phis_record = []
    for batch in range(1000):
        e = RandomWalkEnv(Qs) #BanditHardAndSparse2() #DeceptiveBanditOneHigh10() #Env([1,0,0,0])
        agent = Learner(e, params)
        (rewards, phis) = agent.run(100)
        rewards_record.append(rewards)
        phis_record.append(phis)
    rewards_by_agent.append(rewards_record)
    means = np.array(rewards_record).mean(axis=0)
    stds = np.array(rewards_record).std(axis=0)
    axs[1].plot(means, label=str(phi))
    axs[1].fill_between(np.arange(100), means+stds, means-stds, alpha=0.2)

params = {'t_mix': 3.0, 't_img': 3.0, 'alpha': 0.1}
rewards_record = []
phis_record = []
for batch in range(1000):
    e = RandomWalkEnv(Qs) #BanditHardAndSparse2() #DeceptiveBanditOneHigh10() #Env([1,0,0,0])
    agent = Learner(e, params)
    (rewards, phis) = agent.run(100)
    rewards_record.append(rewards)
    phis_record.append(phis)
rewards_by_agent.append(rewards_record)
means = np.array(rewards_record).mean(axis=0)
stds = np.array(rewards_record).std(axis=0)
axs[1].plot(means, label='auto', c='k', linestyle='dashed')
axs[1].fill_between(np.arange(100), means+stds, means-stds, color='k', alpha=0.1)

#fig.suptitle('Avg. reward over 1,000 runs +- std')
axs[1].set_title('Avg. reward over 1,000 runs +- std')
axs[1].set_ylabel('reward')
axs[1].legend(title='phi setting', loc='upper left')

means = np.array(phis_record).mean(axis=0)
stds = np.array(phis_record).std(axis=0)
axs[2].plot(means, label='phi', c='k')
axs[2].fill_between(np.arange(100), means+stds, means-stds, color='k', alpha=0.1)
axs[2].set_title('Avg. adaptive phi setting over 1,000 runs +- std')
axs[2].set_ylim(0,1)
axs[2].set_ylabel('phi')
axs[2].set_xlabel('trial')
axs[2].legend(loc='upper left')
 
plt.show()


# fig, axs = plt.subplots(3)
# for phi in [0.0, 0.25, 0.5, 0.75, 1.0]:

#     #fig.suptitle('Vertically stacked subplots')
#     axs[0].plot(x, y)
#     axs[1].plot(x, -y)

fig, ax = plt.subplots(1, figsize=(8,4))

batch_means = []
for agent_data in rewards_by_agent:
    avg_rewards = []
    for batch_rewards in agent_data:
        avg_rewards.append(np.array(batch_rewards).mean())
    batch_means.append(np.array(avg_rewards).mean())

ax.bar(np.arange(6), batch_means)
ax.set_xticklabels(labels=['0.0', '0.0', '0.25', '0.5', '0.75', '1.0', 'auto'])
ax.set_xlabel('Phi setting')
ax.set_ylabel('Average experiment-averaged reward')

plt.show()





