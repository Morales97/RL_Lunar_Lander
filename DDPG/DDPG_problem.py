# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent, DDPGAgent, ActorNeuralNetwork
from collections import deque
import pdb

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()
env.seed(0)
np.random.seed(0)

# Parameters
N_episodes = 300                                # Number of episodes to run for training
discount_factor = 1                         # Value of gamma
n_ep_running_average = 50                       # Running average of 50 episodes
dim_action = len(env.action_space.high)         # dimensionality of the action is 2 (main engine and direction engine)
dim_state = len(env.observation_space.high)     # dimensionality of the state is 8 (x pos, y pos, x vel, y vel, angle, angular velocity, bool contact left, bool contact right)
erb_size = 30000                                # Size of buffer
batch_size = 64                                 # Size of batch sampled from buffer

# Paramters of DDPG Agent
lr_critic = 5e-4               # Learning rate of critic
lr_actor = 5e-5                # Learning rate of actor - should be lesser than critic's: first learn value function to be able to learn policy
tau = 0.005                    # Soft update coefficient 
delay = 2                      # Actor network update delay
noise_mu = 0.15                # Noise correlation with previous noise sample
noise_sigma = 0.2              # Noise variance

# Reward
episode_reward_list = []       # Used to save episodes reward
episode_number_of_steps = []

# Experience Replay Buffer initialization
ERB = deque(maxlen = erb_size)

# Agent initialization
agent_rand = RandomAgent(dim_action)
agent = DDPGAgent(dim_action, dim_state, discount_factor, lr_critic, lr_actor, tau, delay, noise_mu, noise_sigma)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

def fill_ERB():
    while len(ERB) < erb_size:
        done = False
        state = env.reset()
        while not done:
            
            # Take a random action
            action = agent_rand.forward(state)
            # Get next state and reward. The done variable will be True if you reached the goal position, False otherwise
            next_state, reward, done, _ = env.step(action)

            # Append last experience to buffer
            ERB.append((state, action, reward, next_state, done))

            # Update state for next iteration
            state = next_state
        env.close()

def train_agent():
    for i in EPISODES:
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = agent.forward(state)

            # Get next state and reward. The done variable will be True if you reached the goal position, False otherwise
            next_state, reward, done, _ = env.step(action)

            if type(agent) is DDPGAgent:
                # Append last experience to buffer
                ERB.append((state, action, reward, next_state, done))

                #if i % 30 == 0 or i == 299:
                    #env.render()

                if len(ERB) >= batch_size:
                    # Random batch
                    batch_idxs = np.random.choice(
                        len(ERB),
                        size = batch_size,
                        replace = False)
                    batch = [ERB[index] for index in batch_idxs]

                    # Train critic network
                    agent.train_critic(batch)

                    # Every d timesteps, Train actor network and Update target networks (soft update)
                    if t % delay == 0:
                        agent.train_actor(batch)
                        agent.update_targets()

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))


    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()

    # Save networks
    if type(agent) is DDPGAgent:# and input('Input Anything to Save or Enter to Exit...'):
        agent.save_networks()

def play_and_render():
    model = torch.load('neural-network-2-actor.pth')
    #model = ActorNeuralNetwork(8,2)
    #model.load_state_dict(torch.load('sd-nn-actor.pth'))
    done = False
    state = env.reset()
    total_episode_reward = 0.
    while not done:
        env.render()
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        action = model(torch.tensor([state]))[0]
        next_state, reward, done, _ = env.step(action.detach().numpy())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
    print(total_episode_reward)
    env.close()
    
def train():
    fill_ERB()
    train_agent()

# CHOOSE EITHER TO RENDER A GAME OR TO TRAIN
play_and_render()
#train()