#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 2022

@author: Jack Burgess
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Environment class
class Env:
    
    # initialization method
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.rng = np.random.default_rng() # add random seed here if wanted
    
    # generate environmental sample
    def sample(self):
        X = self.rng.random(self.sample_size) # X ~ Unif[0,1)
        p = 1 / (1 + np.exp(-10 * (X - 0.5))) # p(x is 1) follows logistic
        Y = self.rng.random(self.sample_size) < p # generate labels from probs
        Y = Y.astype(int) # cast bool labels to integer type
        return (X, Y)


# Learner class
class Learner:
    
    # initialization method
    def __init__(self, env):
        self.env = env
        # instantiate models
        self.m_e = LinearRegression()
        self.m_i = LinearRegression()
        self.m_c = LinearRegression()
        # initialize models with random data
        self.m_e.fit(self.env.rng.random(self.env.sample_size).reshape(-1, 1),
                      self.env.rng.choice(2, self.env.sample_size))
        self.m_i.fit(self.env.rng.random(self.env.sample_size).reshape(-1, 1),
                      self.env.rng.choice(2, self.env.sample_size))
        self.m_c.fit(self.env.rng.random(self.env.sample_size).reshape(-1, 1),
                      self.env.rng.choice(2, self.env.sample_size))
    
    # run learner on environment
    def run(self, params):
        
        m_e_scores = []
        m_i_scores = []
        m_c_scores = []
        
        for run in range(params['n_runs']):
            (X_w, Y_w) = self.env.sample()
            #print(self.evaluate(X_w, Y_w)) # print current score
            
            m_e_scores.append(self.m_e.score(X_w.reshape(-1, 1), Y_w))
            m_i_scores.append(self.m_i.score(X_w.reshape(-1, 1), Y_w))
            m_c_scores.append(self.m_c.score(X_w.reshape(-1, 1), Y_w))
            
            self.update(X_w, Y_w, params)
            
        return (m_e_scores, m_i_scores, m_c_scores)
    
    # evaluate current state
    def evaluate(self, X_w, Y_w):
        return self.m_c.score(X_w.reshape(-1, 1), Y_w)
    
    # update current state
    def update(self, X_w, Y_w, params):
        # calculate environment and imagination model losses
        Y_hat_e = self.m_e.predict(X_w.reshape(-1, 1))
        loss_e = np.sum( (Y_w - Y_hat_e) ** 2)
        Y_hat_i = self.m_i.predict(X_w.reshape(-1, 1))
        loss_i = np.sum( (Y_w - Y_hat_i) ** 2)
        phi = loss_e / (loss_e + loss_i) # relative loss proportion
        
        #print("phi = " + str(phi))
        #print(self.m_e.score(X_w.reshape(-1, 1), Y_w))
        #print(self.m_i.score(X_w.reshape(-1, 1), Y_w))
        #print(self.m_c.score(X_w.reshape(-1, 1), Y_w))
        #print()
        
        # train environment model on environment data
        self.m_e.fit(X_w.reshape(-1, 1), Y_w)
        
        # train imagination model on imagined data
        X_i = self.env.rng.random(self.env.sample_size) # X ~ Unif[0,1)
        Y_i = self.m_c.predict(X_w.reshape(-1, 1)) # imagine from combo model
        self.m_i.fit(X_i.reshape(-1, 1), Y_i)
        
        theta_e = (self.m_e.coef_, self.m_e.intercept_)
        theta_i = (self.m_i.coef_, self.m_i.intercept_)
        theta_c = (self.m_c.coef_, self.m_c.intercept_)
        
        if 'phi' in params:
            phi = params['phi']
        
        theta_x = (theta_e[0] * (1 - phi) + theta_i[0] * phi,
                   theta_e[1] * (1 - phi) + theta_i[1] * phi)
        
        gam = params['gam']
        
        theta_c = (theta_x[0] * (1 - gam) + theta_c[0] * gam,
                   theta_x[1] * (1 - gam) + theta_c[1] * gam)
        
        (self.m_c.coef_, self.m_c.intercept_) = theta_c
        
        #print(theta_e)
        #print(sample_w)



plt.figure()
plt.xlabel('batch')
plt.ylabel('R ^ 2')
plt.ylim(-1,1)

for gam in np.arange(0, 1, 0.25):
    params = {'n_runs': 30, 'phi': 0, 'gam': gam}
    
    e = Env(sample_size=10)
    l = Learner(e)
    results = l.run(params)
    
    
    #plt.plot(results[0], label='M_e')
    plt.plot(results[2], label=('gam = ' + str(gam)))
    plt.legend()