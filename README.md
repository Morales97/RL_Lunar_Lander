# Reinforcement Learning: Lunar Lander
Lunar Lander environment by openAI's gym (https://gym.openai.com/envs/LunarLander-v2/) solved using 3 different algorithms: DQN, DDPG and PPO.

This project belongs to the course EL2805 Reinforcement Learning at KTH.

## Deep Q-Network (DQN)
In this version of the problem we use a discrete action space with 4 actions (left, right, main engine or do nothing). 
Q-learning finds the optimal policy by learning the Q-function, aka (action, state)-value function. The Q-function returns the expected future reward if action a is selected from state s.

However, due to the very large state space of the Lunar Lander environment, learning Q(s,a) for every state s is not feasible. 
Instead, in DQN we approximate the Q-function with a neural network, which takes the state as input and returns the Q-value for each of the 4 actions.
The weights and biases of the NN are the parameters being learned, and not the actual Q-function for every state. This reduces the complexity of the algorithm.
<p align="center">
<img src="/DQN/DQN_algorithm.png" width="600"/>
<p/>


## Deep Deterministic Policy Gradient (DDPG)
Now we switch to a version fo the problem with a continuous actions space, which is more realistic and is the usual scenario of most cyber-physical systems such as automonous vehicles.

With a continuous action space, we can no longer use DQN. This is due to the very large complexity of comparing Q-values for all possible actions when dealing with continuous actions. The solution is to use actor-critic algorithms in which we estimate both the critic (Q-function) and the actor (policy). Similarly to DQN, we have a NN to approximate Q-function, but now we also have another NN that approximates a deterministic policy. This policy is used to generate the samples and is periodically updated using the Deterministic Policy Gradient method. 
<p align="center">
<img src="/DDPG/DDPG_algorithm.png" width="600"/>
<p/>

## Proximal Policy Optimization (PPO)
PPO is an attempt to improve the stability of policy gradient methods. It is also an actor-critic algorithm and we will work with a continuous action space.

PPO is a Trust Region algorithm, where the policy learned is a region of actions instead of being deterministic. In this case, the NN of the policy will return mean and variance given the state, and it denotes a gaussian random variable that will be used to select the action. To prevent sudden changes in the policy network, we bound the update with a clipping function.
<p align="center">
<img src="/PPO/PPO_algorithm.png" width="600"/>
<p/>
