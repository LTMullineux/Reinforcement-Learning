# -*- coding: utf-8 -*-
import numpy as np

# =============================================================================
# Class holder for a single-armed bandit
# Arm can either be:
#     - Stationary:
#         The arm has a constant noraml reward distribution N(0,1)
#     - Non-Stationary:
#         The arm has a reward distribution N(mean=0, var=1) but follows a 
#         random walk with gaussian noise N(walkMean=0, walkVar=0.1)
# =============================================================================

class Arm:
    ''' A arm with a normal distribution, non-stationary optional'''
    def __init__(self, mean=0, var=0.1, 
                 isStationary=True, walkMean = 0, walkVar = 0.1):
        self.mean = mean
        self.init_mean = mean
        self.var = var
        self.isStationary = isStationary
        self.pulls = 0
        self.walkMean = walkMean
        self.walkVar = walkVar
        
    def pull(self, size=1):
        ''' Return a random sample(s) from the arm's reward distribution '''
        if size == 1:
            reward = np.random.normal(self.mean, self.var)
            if not self.isStationary:
                self.walk()
            return reward
        else:
            rewards = []
            for i in range(size):
                reward = np.random.normal(self.mean, self.var)
                rewards.append(reward)
                if not self.isStationary:
                    self.walk()
            return np.array(rewards)
    
    def walk(self):
        ''' Increment mean reward as random walk '''
        step = np.random.normal(self.walkMean, self.walkVar)
        self.mean += step
        
    def reset(self):
        ''' reset all class attributes '''
        self.mean = self.init_mean
        self.pulls = 0
