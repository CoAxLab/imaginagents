#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:59:18 2022

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt

# from bandit import DeceptiveBanditOneHigh10
# from bandit import BanditHardAndSparse2
# from bandit import BanditHardAndSparse10
# from bandit import BanditHardAndSparse121
# from bandit import BanditHardAndSparse1000


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
    
    # reset method
    def reset(self):
        self.counter = 0


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
    
    # reset method
    def reset(self):
        self.counter = 0


class BanditEnv:
    
    # initialization method
    def __init__(self, arm_probs):
        self.arm_probs = arm_probs
        Qs = []
        for i in range(100):
            Qs.append((np.random.rand(4) < self.arm_probs).astype(int))
        self.env = RandomWalkEnv(Qs)
        self.action_space = self.env.action_space
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        Qs = []
        for i in range(100):
            Qs.append((np.random.rand(4) < self.arm_probs).astype(int))
        self.env = RandomWalkEnv(Qs)


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
        loss_diffs = []
        exp_losses = []
        img_losses = []
        for trial in range(n_trials):
            (reward, phi, loss_exp, loss_img) = self.perform_trial()
            rewards.append(reward)
            phis.append(phi)
            exp_losses.append(loss_exp)
            img_losses.append(loss_img)
            loss_diffs.append(loss_img - loss_exp)
        # print(self.Q_exp)
        # print(self.Q_img)
        # print(self.Q_mix)
        return (rewards, phis, exp_losses, img_losses)
    
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
        if self.env_steps_taken == 1:
            phi = 0 # phi is 0 for the 1st time step
        elif 'phi' in self.params:
            phi = self.params['phi']
        elif loss_exp + loss_img == 0: # prevent divide by zero error
            phi = 0.5
        else:
            #phi = loss_exp / (loss_exp + loss_img) # relative loss proportion
            
            # TODO - implement logistic phi setting
            phi = 1 / (1 + np.exp(-10 * (loss_exp - loss_img)))
            
            # # all-or-nothing phi
            # if loss_img - loss_exp < 0: #0.05: #0: #0.1:
            #     phi = 1.0
            # else:
            #     phi = 0.0
        
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
        return (reward, phi, loss_exp, loss_img)


def run_and_plot_phis_exp(env, arm_vals, task_ylabel, portion='all'):

    n_trials = 100    
    n_batches = 1000
    if portion == 'all':
        start = 0
        stop = n_trials
    else:
        start = portion[0]
        stop  = portion[1]
    
    fig, axs = plt.subplots(3, sharex=True, figsize=(8,10))
    
    axs[0].plot(np.arange(start, stop), np.array(arm_vals)[start:stop,0], c='C6', lw=4, alpha=0.6, label='A')
    axs[0].plot(np.arange(start, stop), np.array(arm_vals)[start:stop,1], c='C5', lw=4, alpha=0.6, label='B')
    axs[0].plot(np.arange(start, stop), np.array(arm_vals)[start:stop,2], c='C8', lw=4, alpha=0.6, label='C')
    axs[0].plot(np.arange(start, stop), np.array(arm_vals)[start:stop,3], c='C9', lw=4, alpha=0.6, label='D')
    axs[0].set_title('4-armed bandit task structure')
    axs[0].set_ylabel(task_ylabel) #'Reward') #'P(reward)')
    axs[0].legend(title='arm', loc='upper right')
    
    rewards_by_agent = []
    
    for phi in [0.0, 0.25, 0.5, 0.75, 1.0]:
        params = {'t_mix': 3, 't_img': 3, 'alpha': 0.1, 'phi': phi}
        rewards_record = []
        phis_record = []
        for batch in range(n_batches):
            env.reset()
            agent = Learner(e, params)
            (rewards, phis, exp_losses, img_losses) = agent.run(n_trials)
            rewards_record.append(rewards)
            phis_record.append(phis)
        rewards_by_agent.append(rewards_record)
        means = np.array(rewards_record).mean(axis=0)[start:stop]
        stds = np.array(rewards_record).std(axis=0)[start:stop]
        cis = 1.96 * stds / np.sqrt(n_batches)
        axs[1].plot(np.arange(start, stop), means, label=str(phi))
        axs[1].fill_between(np.arange(start, stop), means+stds, means-stds, alpha=0.2)
    
    params = {'t_mix': 3, 't_img': 3, 'alpha': 0.1}
    rewards_record = []
    phis_record = []
    #diff_record = []
    exp_losses_record = []
    img_losses_record = []
    for batch in range(n_batches):
        env.reset()
        agent = Learner(env, params)
        (rewards, phis, exp_losses, img_losses) = agent.run(n_trials)
        rewards_record.append(rewards)
        phis_record.append(phis)
        #diff_record.append(loss_diffs)
        exp_losses_record.append(exp_losses)
        img_losses_record.append(img_losses)
    rewards_by_agent.append(rewards_record)
    means = np.array(rewards_record).mean(axis=0)[start:stop]
    stds = np.array(rewards_record).std(axis=0)[start:stop]
    cis = 1.96 * stds / np.sqrt(n_batches)
    axs[1].plot(np.arange(start, stop), means, label='auto', c='k', linestyle='dashed')
    axs[1].fill_between(np.arange(start, stop), means+stds, means-stds, color='k', alpha=0.1)
    axs[1].set_title('Avg. reward over 1,000 runs (+- SD)')
    axs[1].set_ylabel('Reward')
    axs[1].legend(title='phi setting', loc='upper right')
    
    means = np.array(phis_record).mean(axis=0)[start:stop]
    stds = np.array(phis_record).std(axis=0)[start:stop]
    cis = 1.96 * stds / np.sqrt(n_batches)
    axs[2].plot(np.arange(start, stop), means, label='phi', c='k')
    axs[2].fill_between(np.arange(start, stop), means+stds, means-stds, color='k', alpha=0.1)
    axs[2].set_title('Avg. adaptive phi setting over 1,000 runs (+- SD)')
    axs[2].set_ylim(-0.05,1.05)
    axs[2].set_ylabel('phi')
    axs[2].set_xlabel('Trial')
     
    plt.show()
    
    # fig, ax = plt.subplots(1, figsize=(8,4))
    
    # means = np.array(diff_record).mean(axis=0)[start:stop]
    # stds = np.array(diff_record).std(axis=0)[start:stop]
    # cis = 1.96 * stds / np.sqrt(n_batches)
    # ax.plot(np.arange(start, stop), means, label='phi', c='k')
    # ax.fill_between(np.arange(start, stop), means+stds, means-stds, color='k', alpha=0.1)
    # ax.set_title('Avg. loss difference (L_I - L_E) over 1,000 runs (+- SD)')
    # #ax.set_ylim(0,1)
    # ax.set_ylabel('loss difference (L_I - L_E)')
    # ax.set_xlabel('Trial')
    # ax.axhline(y=0)
    
    # plt.show()
    
    fig, axs = plt.subplots(2, figsize=(8,7), sharex=True)
    
    exp_means = np.array(exp_losses_record).mean(axis=0)[start:stop]
    img_means = np.array(img_losses_record).mean(axis=0)[start:stop]
    exp_stds = np.array(exp_losses_record).std(axis=0)[start:stop]
    img_stds = np.array(img_losses_record).std(axis=0)[start:stop]
    exp_cis = 1.96 * exp_stds / np.sqrt(n_batches)
    img_cis = 1.96 * img_stds / np.sqrt(n_batches)
    axs[0].plot(np.arange(start, stop), exp_means, label='L_E')#, c='k')
    axs[0].fill_between(np.arange(start, stop), exp_means+exp_stds, exp_means-exp_stds, alpha=0.1) #, label='L_E') #color='k', )
    axs[0].plot(np.arange(start, stop), img_means, label='L_I')#, c='k')
    axs[0].fill_between(np.arange(start, stop), img_means+img_stds, img_means-img_stds, alpha=0.1) #, label='L_I') #color='k', )
    axs[0].set_title('Avg. L_I vs. L_E over 1,000 runs (+- SD)')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right')
    axs[1].plot(np.arange(start, stop), 1 / (1 + np.exp(-10 * (exp_means - img_means))), c='k', label='sigmoid of difference')
    axs[1].set_title('Logistic of difference')
    axs[1].set_ylabel('phi')
    axs[1].set_xlabel('Trial')
    #axs[1].axhline(y=0)
    axs[1].set_ylim(-0.05,1.05)
    
    plt.show()
    
    fig, ax = plt.subplots(1, figsize=(8,4))
    
    batch_means = []
    batch_stdvs = []
    for agent_data in rewards_by_agent:
        avg_rewards = []
        for batch_rewards in agent_data:
            avg_rewards.append(np.array(batch_rewards)[start:stop].mean())
        batch_means.append(np.array(avg_rewards).mean())
        batch_stdvs.append(np.array(avg_rewards).std())
    
    # how long to make error bars (one side of the bar)
    ci_err_95 = 1.96 * np.array(batch_stdvs) / np.sqrt(n_batches)
    ax.bar(np.arange(6), batch_means, yerr=ci_err_95, color='lightgreen')
    ax.set_xticklabels(labels=['0.0', '0.0', '0.25', '0.5', '0.75', '1.0', 'auto'])
    ax.set_xlabel('Phi setting')
    
    if portion == 'all':
        ax.set_ylabel('Average reward')
        ax.set_title('Average reward across experiment by agent (95% CI)')
    else:
        ax.set_ylabel('Average reward')
        ax.set_title(f'Average reward by agent, across trials {start} up to {stop} (95% CI)')
    
    plt.show()


##### EXPERIMENTS #####

# # switch experiment
# e = Env([1,0,0,0])
# run_and_plot_phis_exp(env=e, arm_vals=[[1,0,0,0]]*50+[[0,0,0,1]]*50, task_ylabel='Reward')


# generate random walk reward schedules for drift experiment
A_init = np.random.rand()
Qs = [np.array([A_init, 0.2, 0.3, 1 - A_init])]
#start = np.random.rand(4)
for i in range(99):
    drift = np.random.normal(0, 0.25) #,0.1 #, 4)
    Qs.append(Qs[-1] + [drift, 0, 0, -1 * drift])
    if Qs[-1][0] > 1:
        Qs[-1][0] = 1
        Qs[-1][3] = 0
    if Qs[-1][0] < 0:
        Qs[-1][0] = 0
        Qs[-1][3] = 1

e = RandomWalkEnv(Qs)
run_and_plot_phis_exp(env=e, arm_vals=Qs, task_ylabel='Reward')


# run very sparse random reward schedule experiment
arm_reward_probs = [0.01, 0.02, 0.01, 0.02]
e = BanditEnv(arm_probs=arm_reward_probs)
run_and_plot_phis_exp(env=e, arm_vals=[arm_reward_probs]*100, task_ylabel='P(Reward=1)')


# generate deceptive reward schedules and run experiment
Qs = []
for i in range(20):
    Qs.append(np.array([.2,.9,.2,.2]))
for i in range(10):
    Qs.append(np.array([.2,.1,.2,.2]))
for i in range(70):
    Qs.append(np.array([.2,.9,.2,.2]))

e = RandomWalkEnv(Qs)
run_and_plot_phis_exp(env=e, arm_vals=Qs, task_ylabel='Reward')
run_and_plot_phis_exp(env=e, arm_vals=Qs, task_ylabel='Reward', portion=(15,35))
#run_and_plot_phis_exp(env=e, arm_vals=Qs, task_ylabel='Reward', portion=(20,25))


# # switch experiment (right after switch)
# e = Env([1,0,0,0])
# run_and_plot_phis_exp(env=e,
#                       arm_vals=[[1,0,0,0]]*50+[[0,0,0,1]]*50,
#                       task_ylabel='Reward',
#                       portion=(50,55))


# # switch experiment (right around switch)
# e = Env([1,0,0,0])
# run_and_plot_phis_exp(env=e,
#                       arm_vals=[[1,0,0,0]]*50+[[0,0,0,1]]*50,
#                       task_ylabel='Reward',
#                       portion=(45,55))


# # generate continual switch schedules and run experiment
# Qs = []
# for _ in range(5):
#     for i in range(10):
#         Qs.append(np.array([1,0,0,0]))
#     for i in range(10):
#         Qs.append(np.array([0,0,0,1]))

# e = RandomWalkEnv(Qs)
# run_and_plot_phis_exp(env=e, arm_vals=Qs, task_ylabel='Reward')


# # generate continual switch schedules and run experiment
# Qs = []
# for _ in range(2):
#     for i in range(20):
#         Qs.append(np.array([1,0,0,0]))
#     for i in range(20):
#         Qs.append(np.array([0,0,0,1]))
# for i in range(20):
#         Qs.append(np.array([1,0,0,0]))
        
# e = RandomWalkEnv(Qs)
# run_and_plot_phis_exp(env=e, arm_vals=Qs, task_ylabel='Reward')

fig, ax = plt.subplots(1, figsize=(8,4))
x = np.arange(-1,1,.001)
y = 1 / (1 + np.exp(-10 * (x)))
ax.plot(x, y)
ax.set_title('Adaptive phi function')
ax.set_xlabel('loss_exp - loss_img')
ax.set_ylabel('1 / (1 + exp(-10 * (loss_exp - loss_img)))')
plt.show()