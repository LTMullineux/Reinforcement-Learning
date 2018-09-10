# -*- coding: utf-8 -*-


from arm import *
from bandit import *
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# ----------------------------------------------------------------------------- 
# Base class for all bandits to derive from
class BanditAgent(object):
    ''' Simple class holder for a n-bandit agent '''
    def __init__(self, n, bandit, epsilon=0.05, hot_start=0.0):
        self.n = n
        self.epsilon = epsilon
        self.Q = np.ones(self.n) * hot_start
        self.bandit = bandit

    def getAction(self):
        raise NotImplementedError
    
    def optimalAction(self):
        ''' Returns the bandit with highest mean value '''
        means = [a.mean for a in self.bandit.arms]
        return np.argmax(means)
           
    def pull(self):
        ''' Agent pulls one bandit using eps-greedy action '''
        action = self.getAction()
        reward = self.bandit.arms[action].pull()
        return action, reward

    def learn(self, num_pulls=1000, return_training=True):
        ''' Agent learns to play game over a number of pulls on the bandits '''
        raise NotImplementedError
        
    def reset(self):
        ''' reset all class attributes '''
        raise NotImplementedError

# -----------------------------------------------------------------------------
class RandomBandit(BanditAgent):
    '''random acting agent class '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getAction(self):
        ''' returns the action to take with eps-greedy policy '''
        return np.random.choice(range(self.n))
        
    def learn(self, num_pulls=1000, return_training=True):
            ''' Agent learns game over a number of pulls on arms '''
            # Holders for avg rewards and optimal actions
            avg_rewards = []
            avg_r = 0
            optimal_actions = []
            opt_a = 0
            
            # Learn bandit game
            for i in range(num_pulls):
                i += 1
                optimal_action = self.optimalAction()
                action, reward = self.pull()
                
                avg_r = avg_r + (reward - avg_r)/i
                avg_rewards.append(avg_r)
                
                opt_a = opt_a + ((action == optimal_action) - opt_a)/i
                optimal_actions.append(opt_a)
                
            if return_training:
                return np.array(avg_rewards), np.array(optimal_actions)
            
    def reset(self):
        ''' reset all class attributes '''
        pass
    
# -----------------------------------------------------------------------------
# class for agent with simple eps-greedy strategy
# ... (stationary reward distributions)
class SimpleBandit(BanditAgent):
    '''stationary bandit class, averaging rewards for actions per visit'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = np.zeros((self.n))

    def getAction(self):
        ''' returns the action to take with eps-greedy policy '''
        r = np.random.random()
        if r < self.epsilon:
            return np.random.choice(range(self.n))
        else:
            return np.argmax(self.Q) 

    def learn(self, num_pulls=1000, return_training=True):
            ''' Agent learns game over a number of pulls on arms '''
            # Holders for avg rewards and optimal actions
            avg_rewards = []
            avg_r = 0
            optimal_actions = []
            opt_a = 0
            
            # Learn bandit game
            for i in range(num_pulls):
                i += 1
                optimal_action = self.optimalAction()
                action, reward = self.pull()
                
                old_Q = self.Q[action]
                self.N[action] += 1
                self.Q[action] += (reward - old_Q) / self.N[action]
                
                avg_r = avg_r + (reward - avg_r)/i
                avg_rewards.append(avg_r)
                
                opt_a = opt_a + ((action == optimal_action) - opt_a)/i
                optimal_actions.append(opt_a)
                
            if return_training:
                return np.array(avg_rewards), np.array(optimal_actions)
            
    def reset(self):
        ''' reset all class attributes '''
        self.Q = np.zeros((self.n))
        self.N = np.zeros((self.n))
        
# -----------------------------------------------------------------------------
# class for agent using exponential reward weighting ...
# ... (non-stationary reward distributions)
class WeightedBandit(BanditAgent):
    '''stationary bandit class, averaging rewards for actions per visit'''
    def __init__(self, alpha=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        
    def getAction(self):
        ''' returns the action to take with eps-greedy policy '''
        r = np.random.random()
        if r < self.epsilon:
            return np.random.choice(range(self.n))
        else:
            return np.argmax(self.Q) 

    def learn(self, num_pulls=1000, return_training=True):
            ''' Agent learns game over a number of pulls on bandits '''
            # Holders for avg rewards and optimal action thru time
            avg_rewards = []
            avg_r = 0
            optimal_actions = []
            opt_a = 0
            
            # Learn bandit game
            for i in range(num_pulls):
                i += 1
                optimal_action = self.optimalAction()
                action, reward = self.pull()
                
                old_Q = self.Q[action]
                self.Q[action] += self.alpha * (reward - old_Q)
                
                # track avg reward and optimal actions
                avg_r = avg_r + (reward - avg_r)/i
                avg_rewards.append(avg_r)
                
                opt_a = opt_a + ((action == optimal_action) - opt_a)/i
                optimal_actions.append(opt_a)
                
            if return_training:
                return np.array(avg_rewards), np.array(optimal_actions)
            
    def reset(self):
        ''' reset all class attributes '''
        self.Q = np.zeros((self.n))
        self.N = np.zeros((self.n))
        
        
# -----------------------------------------------------------------------------
# Upper-confidence-bound bandit agent class
class UCBBandit(BanditAgent):
    '''Upper-confidence-bound bandit agent class'''
    def __init__(self, c=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = np.zeros(self.n)
        self.c = c
        self.t = 1

    def getAction(self):
        ''' action according to upper-confidence bound, max over actions '''
        # Try all actions at least once before the rest
        for action in range(self.n):
            if self.N[action] == 0:
                return action
        
        # Once all tried, choose the action with highest UCB value
        else:
            exploration_bonus = self.c + np.sqrt(np.log(self.t) / self.N)
            ucb_values = self.Q + exploration_bonus
            return np.argmax(ucb_values)

    def learn(self, num_pulls=1000, return_training=True):
            ''' Agent learns game over a number of pulls on arms '''
            # Holders for avg rewards and optimal actions
            avg_rewards = []
            avg_r = 0
            optimal_actions = []
            opt_a = 0
            
            # Learn bandit game
            for i in range(num_pulls):
                i += 1
                self.t += 1
                optimal_action = self.optimalAction()
                action, reward = self.pull()
                
                old_Q = self.Q[action]
                self.N[action] += 1
                self.Q[action] += (reward - old_Q) / self.N[action]
                
                avg_r = avg_r + (reward - avg_r)/i
                avg_rewards.append(avg_r)
                
                opt_a = opt_a + ((action == optimal_action) - opt_a)/i
                optimal_actions.append(opt_a)
                
            if return_training:
                return np.array(avg_rewards), np.array(optimal_actions)
            
    def reset(self):
        ''' reset all class attributes '''
        self.Q = np.zeros(self.n)
        self.N = np.zeros(self.n)
        self.t = 0
        
        
# -----------------------------------------------------------------------------
# Gradient bandit agent class using softmax policy
class GradientBandit(BanditAgent):
    ''' Gradient bandit agent class'''
    def __init__(self, alpha=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.H = np.zeros(self.n)
        self.actions = range(self.n)
        self.alpha = alpha

    def getActionDist(self):
        ''' softmax action distribution, with max to prevent overflow '''
        action_dist = np.exp(self.H - np.max(self.H)) \
                        / np.sum(np.exp(self.H - np.max(self.H)), axis=0)
        return action_dist
    
    def getAction(self):
        ''' chose action proportional to softmax action distribution'''
        action = np.random.choice(self.actions, p = self.getActionDist())
        return action

    def learn(self, num_pulls=1000, return_training=True):
            ''' Agent learns game over a number of pulls on arms '''
            # Holders for avg rewards and optimal actions
            avg_rewards = []
            avg_r = 0
            optimal_actions = []
            opt_a = 0
            
            # Learn bandit game
            for i in range(num_pulls): 
                i += 1
                optimal_action = self.optimalAction()
                action, reward = self.pull()
                action_dist = self.getActionDist()

                # update action distribution H
                actions_not_taken = self.actions != action
                self.H[action] += self.alpha * (reward - avg_r) \
                                        * (1 - action_dist[action])           
                self.H[actions_not_taken] -= self.alpha * (reward - avg_r) \
                                                * action_dist[actions_not_taken]
                
                # update running averages
                avg_r = avg_r + (reward - avg_r)/i
                avg_rewards.append(avg_r)
                
                opt_a = opt_a + ((action == optimal_action) - opt_a)/i
                optimal_actions.append(opt_a)
                
            if return_training:
                return np.array(avg_rewards), np.array(optimal_actions)
            
    def reset(self):
        ''' reset all class attributes '''
        self.H = np.zeros((self.n))
        
        
# -----------------------------------------------------------------------------
# Bayesian bandit class assuming all arms have variance 1 known
class BayesianBandit(BanditAgent):
    ''' Bayesian bandit agent class with normal priors'''
    def __init__(self, alpha=0.1, min_try=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = np.zeros(self.n)
        self.vars = np.ones(self.n)
        self.actions = range(self.n)
        self.N = np.ones((self.n))
        self.arm_rewards = np.zeros(self.n)
        self.alpha = alpha
        self.min_try = min_try
    
    def getAction(self):
        ''' chose action proportional to softmax action distribution'''
        # Try all actions at least min_try times before the rest
        for action in range(self.n):
            if self.N[action] <= self.min_try:
                return action
            
        # Then begin to sample from posteriors, choosing the highest mean
        else:
            sample_posterior = np.array([np.random.normal(mu, var) for 
                                         mu,var in zip(self.means, self.vars)])
            return np.argmax(sample_posterior)

    def learn(self, num_pulls=1000, return_training=True):
            ''' Agent learns game over a number of pulls on arms '''
            # Holders for avg rewards and optimal actions
            avg_rewards = []
            avg_r = 0
            optimal_actions = []
            opt_a = 0
            
            # Learn bandit game
            for i in range(num_pulls):
                i += 1
                optimal_action = self.optimalAction()
                action, reward = self.pull()
                
                # update posterior mean and variance
                self.N[action] += 1
                self.arm_rewards[action] += reward
                self.means[action] = self.alpha * self.arm_rewards[action].sum()\
                                        / self.N[action]
                self.vars[action] = 1 / self.N[action]

                # update running averages
                avg_r = avg_r + (reward - avg_r)/i
                avg_rewards.append(avg_r)
                
                opt_a = opt_a + ((action == optimal_action) - opt_a)/i
                optimal_actions.append(opt_a)
                
            if return_training:
                return np.array(avg_rewards), np.array(optimal_actions)
            
    def reset(self):
        ''' reset all class attributes '''
        self.N = np.ones((self.n))
        self.means = np.zeros(self.n)
        self.vars = np.ones(self.n)
        self.arm_rewards = np.zeros(self.n)
        

# -----------------------------------------------------------------------------
# Method for list of agents to learn simultaneously on the same bandit
def learnMultiEpisodes(agent_list, n=10,
                       num_episodes=1000, episode_len=1000,
                       isStationary=True):
    ''' 
        Runs multiple episodes for a list of agents
        Same bandit reward distribution used for each run
        Return the averaged rewards and optimal decisions per episode
    '''
    
    # Holders for the statistics for all agents
    num_agents = len(agent_list)
    avg_rewards = [np.zeros(episode_len) for i in range(num_agents)]
    optimal_actions = [np.zeros(episode_len) for i in range(num_agents)]

    # run thru episodes
    for episode in range(num_episodes):
        if episode % 100 == 0:
            print("Executing episode: " + str(episode))
            
        # Create bandit for agent to learn from and overwrite for each bandit
        # Ensures same bandit for each agent in each episode
        nBandit = nArmedBandit(n=n, isStationary=isStationary)
            
        # Loop through each agent
        for agent in range(len(agent_list)):
            # overwrite bandit for each agent and get rewards/optimal actions
            agent_list[agent].bandit = nBandit
            avg_r, opt_a = agent_list[agent].learn(num_pulls = episode_len)
            
            avg_rewards[agent] += avg_r
            optimal_actions[agent] += opt_a
            
            # reset the the bandit after each agent episode and the agent
            nBandit.reset()
            agent_list[agent].reset()
            
    # Average the total rewards for each agents rewards/optimal actions
    avg_rewards = [avg_r/num_episodes for avg_r in avg_rewards]
    optimal_actions = [opt_a/num_episodes for opt_a in optimal_actions] 

    return avg_rewards, optimal_actions  


# -----------------------------------------------------------------------------
# Method for plotting different bandits learning
def plotAgentsLearning(avg_rewards, optimal_actions, bandit_names):
    ''' Plots the average rewards and optimal action % for bandits '''
    episode_len = len(avg_rewards[0])
    t = np.arange(episode_len)
    
    fig, ax = plt.subplots(2, sharex = True)
    for i in range(len(bandit_names)):
        ax[0].plot(t, avg_rewards[i])
        ax[1].plot(t, optimal_actions[i])

    # Custom y-tick and axis labels
    ax[1].set_xlabel("Time Steps $t$")
    ax[0].set_ylabel("Average\n Reward")
    ax[1].set_ylabel("%\nOptimal\nAction")
    ax[1].set_yticks([i for i in np.arange(0, 1.2, 0.2)])
    ax[1].set_yticklabels([str(i)+"%" for i in np.arange(0, 120, 20)])
    
    # Show legend
    ax[0].legend(bandit_names, loc="lower right")
    ax[1].legend(bandit_names, loc="lower right")
    
    plt.show()
    
    
    

