# -*- coding: utf-8 -*-

# =============================================================================
# Class holder for n-armed bandit. Each arm of the bandit has a normal
# reward distribution, with mean sampled from mean_range, and variance from
# var_range. The rewards can be non-stationary with a gaussian 
# random walk, who's parameters are sampled from walkMeans and walkVars.
# =============================================================================

from arm import *
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class nArmedBandit(object):
    ''' n-armed bandit, consisting of n Arms() '''
    def __init__(self, n, mean_range = [-10.0,10.0], var_range=[1.0,1.0],
                 isStationary=True, walkMeans = [0.0,0.0], walkVars = [0.25,0.5]):
        self.n = n
        self.isStationary = isStationary
        self.means = np.random.uniform(mean_range[0], mean_range[1], n)
        self.vars = np.random.uniform(var_range[0], var_range[1], n)
        self.walkMeans = np.random.uniform(walkMeans[0], walkMeans[1], n)
        self.walkVars = np.random.uniform(walkVars[0], walkVars[1], n)
        
        # Create a list of arms
        self.arms = [self.createArm(m, v, isStationary, wm, wv) 
                        for m,v,wm,wv in zip(self.means, self.vars,self.walkMeans, self.walkVars)]
        
    def createArm(self, mean, var, isStationary, walkMean, walkVar):
        ''' create arm with random mean/var given ranges provided '''
        arm = Arm(mean, var, isStationary, walkMean, walkVar)
        return arm
    
    def pullAll(self, size=1):
        ''' Return a random sample from all the bandits' arms '''
        return [a.pull(size) for a in self.arms]
    
    def pull(self, arm_index, size=1):
        ''' Return a random sample from one arm'''
        return self.arms[arm_index].pull(size)
    
    def reset(self):
        ''' reset all class attributes '''
        self.arms = [self.createArm(m, v, self.isStationary, wm, wv) 
                        for m,v,wm,wv in zip(self.means, self.vars,self.walkMeans, self.walkVars)]
        
    def plotBandit(self, sample_size=1000, reset=True):
        ''' 
        Plots the reward distribution of bandit
             - stationary takes sample_size samples
             - non-stationary shows trajectories for sample_size length
        '''
        if self.isStationary:
            rewards = self.pullAll(size=sample_size)
            rewards_mean = [np.sign(r.mean()) for r in rewards]
            
            fig = plt.violinplot(rewards, widths = 1, showmeans=True)
            
            # positive means are green, negative red
            i = 0
            for pc in fig['bodies']:
                if rewards_mean[i] < 0:
                    pc.set_color('red')
                else:
                    pc.set_color('green')
                i += 1

            # Labels
            plt.xticks(np.arange(1, 11, step=1))
            plt.xlabel("Action (Bandit)")
            plt.ylabel("Stationary\n Reward Distrbution")
            plt.axhline(0, c="k")
            plt.show()
            
        else:
            rewards_nonstat = self.pullAll(size=1000)
            t = np.arange(sample_size)
            for arm in rewards_nonstat:
                plt.plot(t, arm)
            plt.xlabel("Time Step $t$")
            plt.ylabel("Non-Stationary\n Reward Distrbution")
            plt.axhline(0, c="k")
            plt.show()

