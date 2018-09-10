# Introduction - N-Armed Bandits
Reinforcement learning is the art of learning how to map situations to actions so as to maximise some defined reward. To do this agents must discover which actions yield the greatest long-term reward, and they must do this with knowledge derived only from feedback from the environment.

To begin with, the most basic environment is evaulated, one where the action chosen has no bearing on the environment. In this situation an agent must learn to play a n-armed bandit, similar to the ones seen in Las Vegas/Monte Carlo/and possibly Blackpool casinos, except with n-arms instead of one. Agents have to explore the reward distributions of the bandit's arms to discover which arm maximises the long-term rewards, and may have to play sub-optimal at times in order to do so.

Different agents with varying strategies are coded in [banditAgents](banditAgents.py) and are evaluated in depth in [the notebook](Bandits.ipynb).
