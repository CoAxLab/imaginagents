#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:06:34 2022

@author: Jack
"""

# Code here taken from Erik Peterson's infomercial GitHub reop

import numpy as np
import gym

from copy import deepcopy
from gym import spaces
from gym.utils import seeding
from itertools import cycle

class DeceptiveBanditEnv(gym.Env):
    """
    n-armed bandit environment, you have to move steps_away to find the best arm.
    Params
    ------
    p_dist : list
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist : list or list or lists
        A list of either rewards (if number) or means and standard deviations (if list) of the payout that bandit has
    """
    def __init__(self, p_dist, r_dist, steps_away=1, max_steps=30):
        if len(p_dist) != len(r_dist):
            raise ValueError(
                "Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError(
                    "Standard deviation in rewards must all be greater than 0")

        if max_steps < (2 * steps_away):
            raise ValueError("max_steps must be greater than 2*steps_away")
        self.p_dist = p_dist
        self.r_dist = r_dist
        self.steps = 0
        self.max_steps = max_steps
        self.steps_away = steps_away
        self.scale = np.concatenate(
            (np.linspace(-1, 0, steps_away), np.linspace(0, 1, steps_away)))

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Sanity
        if self.steps > self.max_steps:
            raise EnvironmentError("Number of steps exceeded max.")

        # Action is in the space?

        action = int(action)
        assert self.action_space.contains(action)

        # Get the reward....
        self.done = True

        reward = 0
        if self.np_random.uniform() < self.p_dist[action]:
            reward = self.r_dist[action]

        # Add deceptiveness. Only the best arms are deceptive.
        if (action in self.best) and (reward != 0):
            try:
                reward *= self.scale[self.steps]
            except IndexError:
                reward *= np.max(self.scale)

            self.steps += 1

        return 0, float(reward), self.done, {}

    def reset(self):
        self.done = False
        return [0]

    def render(self, mode='human', close=False):
        pass


class DeceptiveBanditOneHigh10(DeceptiveBanditEnv):
    """A (0.8, 0.2, 0.2, ...) bandit."""
    def __init__(self):
        self.best = [7]
        self.num_arms = 10

        # Set p(R > 0)
        p_dist = [0.2] * self.num_arms
        p_dist[self.best[0]] = 0.8

        # Set baseline R
        r_dist = [1] * self.num_arms

        DeceptiveBanditEnv.__init__(self,
                                    p_dist=p_dist,
                                    r_dist=r_dist,
                                    steps_away=10,
                                    max_steps=500)


class BanditEnv(gym.Env):
    """
    n-armed bandit environment  
    Params
    ------
    p_dist : list
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist : list or list or lists
        A list of either rewards (if number) or means and standard deviations (if list) of the payout that bandit has
    """
    def __init__(self, p_dist, r_dist):
        if len(p_dist) != len(r_dist):
            raise ValueError(
                "Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError(
                    "Standard deviation in rewards must all be greater than 0")

        self.p_dist = p_dist
        self.r_dist = r_dist

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        #if self.done: #j
        #    raise ValueError("Cannot step, env is done.") #j

        state = 0
        reward = 0
        #self.done = True #j

        if self.np_random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                reward = self.np_random.normal(self.r_dist[action][0],
                                               self.r_dist[action][1])

        return state, reward, None, {} #self.done, {} #j

    def reset(self):
        self.done = False
        return [0]

    def render(self, mode='human', close=False):
        pass


class BanditHardAndSparse2(BanditEnv):
    """A (0.10,0.08,0.08,....) bandit"""
    def __init__(self):
        self.best = [0]
        self.num_arms = 2

        p_dist = [0.01] * self.num_arms
        p_dist[self.best[0]] = 0.02
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditHardAndSparse10(BanditEnv):
    """A (0.10,0.08,0.08,....) bandit"""
    def __init__(self):
        self.best = [7]
        self.num_arms = 10

        p_dist = [0.01] * self.num_arms
        p_dist[self.best[0]] = 0.02
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditHardAndSparse121(BanditEnv):
    """A (0.10,0.08,0.08,....) bandit"""
    def __init__(self):
        self.best = [54]
        self.num_arms = 121

        p_dist = [0.01] * self.num_arms
        p_dist[self.best[0]] = 0.02
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditHardAndSparse1000(BanditEnv):
    """A (0.10,0.08,0.08,....) bandit"""
    def __init__(self):
        self.best = [526]
        self.num_arms = 1000

        p_dist = [0.01] * self.num_arms
        p_dist[self.best[0]] = 0.02
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)
        
        
