# RL_Lunar_Lander
Lunar Lander environment by openAI's gym (https://gym.openai.com/envs/LunarLander-v2/) solved using 3 different algorithms: DQN, DDPG and PPO.

This project belongs to the course EL2805 Reinforcement Learning at KTH.

## Deep Q-Network (DQN)
In this version of the problem we use a discrete action space with 4 actions (left, right, main engine or do nothing). 
Q-learning finds the optimal policy by learning the Q-function, aka (action, state)-value function. The Q-function returns the expected future reward if action a is selected from state s.

However, due to the very large state space of the Lunar Lander environment, learning Q(s,a) for every state s is not feasible. 
Instead, in DQN we approximate the Q-function with a neural network, which takes the state as input and returns the Q-value for each of the 4 actions.
The weights and biases of the NN are the parameters being learned, and not the actual Q-function for every state. This reduces the complexity of the algorithm
<p align="center">
<img src="/DQN/DQN_algorithm.png" width="600"/>
<p/>
