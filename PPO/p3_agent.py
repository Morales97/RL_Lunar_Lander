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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 29th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
from torch import nn
from torch import optim
import pdb
import math

class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)



class PPOAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int, n_state: int, disc, lr_critic, lr_actor, epsilon):
        super(PPOAgent, self).__init__(n_actions)

        # Critic network
        self.critic_nn = CriticNeuralNetwork(n_state)

        # Actor network
        self.actor_nn = ActorNeuralNetwork(n_state, n_actions)

        # Optimizers
        self.critic_opt = optim.Adam(
            self.critic_nn.parameters(),
            lr = lr_critic)
        self.actor_opt = optim.Adam(
            self.actor_nn.parameters(),
            lr = lr_actor)
        
        # Discount factor
        self.disc = disc
        # Epsilon
        self.epsilon = epsilon

    def forward(self, state):
        mu, sigma = self.actor_nn(torch.tensor(state))    # mu and sigma are tensors of the same dimensionality of the action
        mu = mu.detach().numpy()                            
        std = np.sqrt(sigma.detach().numpy())
        a1 = np.random.normal(mu[0], std[0])
        a2 = np.random.normal(mu[1], std[1])
        action = np.clip([a1, a2], -1, 1)
        return action

    def evaluate_2D_gaussian(self, mu, sigma, a):
        prob_0 = torch.pow(2 * np.pi * sigma[:,0], -0.5) * torch.exp(-(a[:,0] - mu[:,0])**2 / (2 * sigma[:,0]))
        prob_1 = torch.pow(2 * np.pi * sigma[:,1], -0.5) * torch.exp(-(a[:,1] - mu[:,1])**2 / (2 * sigma[:,1]))
        return prob_0 * prob_1

    def train(self, buffer, epochs):
        states, actions, rewards, dones = zip(*buffer)
        T = len(dones)

        # Compute target value at each time step
        targets = np.zeros(T)
        targets[T-1] = rewards[T-1]
        for i in reversed(range(T-1)):
            targets[i] = self.disc * targets[i+1] + rewards[i]
        targets = torch.tensor(targets, dtype=torch.float32)    

        # set requires_grad of states, to perform SGD 
        states_grad = torch.tensor(states, requires_grad = True)
        actions = torch.tensor(actions, requires_grad = True)

        old_policy_mu, old_policy_sigma = self.actor_nn(states_grad) # both mu and sigma are Tx2, where T is the length of the episode
        # Vector Tx1 with probability of action under old pi for all the episode
        prob_action_under_old_pi = self.evaluate_2D_gaussian(old_policy_mu, old_policy_sigma, actions).detach()

        # For M epochs
        for _ in range(epochs):
            ## CRITIC'S UPDATE
            # Set gradient to zero
            self.critic_opt.zero_grad()

            # Get value function approximation of critic NN 
            values = self.critic_nn(states_grad).squeeze()   

            # Compute MSE loss
            loss = nn.functional.mse_loss(values, targets)
            # Comput gradient
            loss.backward()

            # Clip gradient norm
            nn.utils.clip_grad_norm_(self.critic_nn.parameters(), max_norm=1)   # TODO: this time is not indicated to use clipping. Try without and asses difference

            # Update critic NN parameters
            self.critic_opt.step()

            ## ACTOR'S UPDATE
            # Set gradient to zero
            self.actor_opt.zero_grad()

            # Get value function from critic NN
            values_no_grad = self.critic_nn(torch.tensor(states)).squeeze()

            # Advantage function
            advantage = targets - values_no_grad

            # Compute ratio between prob with new and old pi
            new_policy_mu, new_policy_sigma = self.actor_nn(states_grad) # both mu and sigma are Tx2, where T is the length of the episode
            prob_action_under_new_pi = self.evaluate_2D_gaussian(new_policy_mu, new_policy_sigma, actions)
            r = prob_action_under_new_pi / prob_action_under_old_pi     # Ratio between new and old policy
            
            # Compute loss
            term_A = r * advantage
            term_B = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantage
            loss = torch.min(term_A, term_B)
            loss = - torch.mean(loss)

            # Compute gradient
            loss.backward()

            # Clip gradient norm
            nn.utils.clip_grad_norm_(self.actor_nn.parameters(), max_norm=1)   # TODO: this time is not indicated to use clipping. Try without and asses difference

            # Update critic NN parameters
            self.actor_opt.step()
            
    def save_networks(self, filename_critic = 'neural-network-3-critic.pth', filename_actor = 'neural-network-3-actor.pth'):
        '''Save networks in working directory'''
        torch.save(self.critic_nn, filename_critic)
        torch.save(self.actor_nn, filename_actor)
        print(f'Saved main critic network as {filename_critic}')
        print(f'Saved main actor network as {filename_actor}')
        return


class CriticNeuralNetwork(nn.Module):
    ''' A feedforward neural network'''
    def __init__(self, input_size):
        super().__init__()
        # params
        hidden_neurons_l1 = 400
        hidden_neurons_l2 = 200
        output_size = 1

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_neurons_l1)
        self.input_layer_activation = nn.ReLU()

        # Create hidden layers
        self.hidden = nn.Linear(hidden_neurons_l1, hidden_neurons_l2)   
        self.hidden_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(hidden_neurons_l2, output_size)

    def forward(self, s):
        ''' Function used to compute the forward pass. Input: state (dim=8). Output: value function (dim=1)'''
        # Compute first layer
        l1 = self.input_layer(s)
        l1 = self.input_layer_activation(l1)

        # Compute hidden layers
        l2 = self.hidden(l1)     
        l2 = self.hidden_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)
        return out


class ActorNeuralNetwork(nn.Module):
    ''' A feedforward neural network'''
    def __init__(self, input_size, n_action):
        super().__init__()
        # params
        hidden_neurons_l1 = 400
        hidden_neurons_l2 = 200
        head_output_size = n_action # output of each head has dimensionality of the actin

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_neurons_l1)
        self.input_layer_activation = nn.ReLU()

        # Create hidden layer, head 1
        self.hidden_head_1 = nn.Linear(hidden_neurons_l1, hidden_neurons_l2)
        self.hidden_activation_head_1 = nn.ReLU()
        # Create hidden layer, head 2
        self.hidden_head_2 = nn.Linear(hidden_neurons_l1, hidden_neurons_l2)
        self.hidden_activation_head_2 = nn.ReLU()

        # Create output layer, head 1 (mu)
        self.output_layer_head_1 = nn.Linear(hidden_neurons_l2, head_output_size)
        self.output_activation_head_1 = nn.Tanh()
        # Create output layer, head 2 (variance)
        self.output_layer_head_2 = nn.Linear(hidden_neurons_l2, head_output_size)
        self.output_activation_head_2 = nn.Sigmoid()

    def forward(self, s):
        ''' Function used to compute the forward pass. Input: state (dim=8). Output: action (dim=2)'''
        # Compute first layer
        l1 = self.input_layer(s)
        l1 = self.input_layer_activation(l1)

        # Compute hidden layers, head 1
        l2_head_1 = self.hidden_head_1(l1)
        l2_head_1 = self.hidden_activation_head_1(l2_head_1)
        # Compute hidden layers, head 2
        l2_head_2 = self.hidden_head_2(l1)
        l2_head_2 = self.hidden_activation_head_2(l2_head_2)

        # Compute output layer, head 1
        out_head_1 = self.output_layer_head_1(l2_head_1)
        out_head_1 = self.output_activation_head_1(out_head_1)   # Apply tanh to constrain action between [-1, 1]
        # Compute output layer, head 2
        out_head_2 = self.output_layer_head_2(l2_head_2)
        out_head_2 = self.output_activation_head_2(out_head_2)   # Apply tanh to constrain action between [-1, 1]
        return (out_head_1, out_head_2)