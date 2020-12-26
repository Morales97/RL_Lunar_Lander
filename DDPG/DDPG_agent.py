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
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
from torch import nn
from torch import optim
import DDPG_soft_updates 
import pdb

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
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)


class DDPGAgent(Agent):
    ''' Agent using DDPG algorithm, child of class Agent '''

    def __init__(self, n_actions: int, n_state: int, disc, lr_critic, lr_actor, tau, delay, mu, sigma):
        super(DDPGAgent, self).__init__(n_actions)

        # Main Networks
        self.main_critic_nn = CriticNeuralNetwork(n_state, n_actions)   # Approximates Q(s,a). Output: Q-value (1-dim)
        self.main_actor_nn = ActorNeuralNetwork(n_state, n_actions)     # Approximates deterministic policy. Output: action vector (2-dim)

        # Target Networks
        self.target_critic_nn = CriticNeuralNetwork(n_state, n_actions)
        self.target_actor_nn = ActorNeuralNetwork(n_state, n_actions)

        # Optimizers
        self.critic_opt = optim.Adam(
            self.main_critic_nn.parameters(),
            lr=lr_critic)
        self.actor_opt = optim.Adam(
            self.main_actor_nn.parameters(),
            lr=lr_actor)

        # Soft update coeff 
        self.tau = tau
        # Target Actor network update delay
        self.d = delay
        # Discount Factor
        self.disc = disc
        # Noise parameters (Ornstein-Uhlenbeck process)
        self.noise_mu = mu
        self.noise_sigma = sigma
        self.noise = np.array([0, 0])

    def forward(self, state):
        action_det = self.main_actor_nn(torch.tensor(state))            
        action = action_det.detach().numpy()                    # This is necessary to transform tensor into array. env(action) can't take tensors
        rand_vector = np.random.normal(0, self.noise_sigma, 2)  
        self.noise = - self.noise_mu * self.noise + rand_vector
        return np.clip(action + self.noise, -1, 1)

    def train_critic(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        # Set gradient to zero
        self.critic_opt.zero_grad() 

        # Compute target vals
        with torch.no_grad():
            next_states = torch.tensor(next_states)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones)==False
            # Compute target deterministic actions given states
            target_det_actions = self.target_actor_nn(next_states)                    # Output should be a tensor with the output of the NN for every state
            # Compute target Q values given states and actions
            target_Q_values = self.target_critic_nn(next_states, target_det_actions)   # Output should be a tensor with the output of the NN for every state-action
            # Compute target vals
            target_vals = rewards + (dones * self.disc * target_Q_values.squeeze())     # squeeze() removes all the dimensions of size 1 (turn from dim [64,1] to [64])

        # Compute Q values
        states = torch.tensor(states, requires_grad=True, dtype = torch.float32)
        actions = torch.tensor(actions, requires_grad=True, dtype = torch.float32)
        Q_values = self.main_critic_nn(states, actions).squeeze()
        # Compute loss function
        loss = nn.functional.mse_loss(Q_values, target_vals)

        # Compute gradient
        loss.backward()

        # Clip gradient norm 
        nn.utils.clip_grad_norm_(self.main_critic_nn.parameters(), max_norm=1)

        # Perform backward pass (backpropagation)
        self.critic_opt.step()
        return

    def train_actor(self, batch):
        ''' How does training work?
        1. Set optimizer gradient to 0
        2. Compute loss function. 
        3. Compute gradient with backward()
            loss.backward() computes dloss/dx for every parameter x which has requires_grad=True
            x are the weights and biases of the NN (I guess...)
            These are accumulated into x.grad for every parameter x, something like: x.grad += dloss/dx
        4. (optional) Clip grad values
        5. Perform a step
            optimizer.step() updates the value of x as: x += -lr * x.grad
        '''
        states, _, _, _, _ = zip(*batch)

        # Set gradient to zero
        self.actor_opt.zero_grad() 

        # Compute deterministic actions given states
        states = torch.tensor(states, requires_grad=True, dtype = torch.float32)
        det_actions = self.main_actor_nn(states)            
        #det_actions.requires_grad = True            # TODO is requires_grad set to True by default, or is this necessary?
        # Compute Q-values given states and actions
        Q_values = self.main_critic_nn(states, det_actions)

        # Compute loss, on which we will perform SGD
        loss = - 1/len(batch) * sum(Q_values)
        
        # Compute gradient
        loss.backward()

        # Clip gradient norm
        nn.utils.clip_grad_norm_(self.main_actor_nn.parameters(), max_norm=1)

        # Perform backward pass (backpropagation)
        self.actor_opt.step()
        return

    def update_targets(self):
        self.target_critic_nn   = DDPG_soft_updates.soft_updates(self.main_critic_nn, self.target_critic_nn, self.tau)
        self.target_actor_nn    = DDPG_soft_updates.soft_updates(self.main_actor_nn,  self.target_actor_nn,  self.tau)
        return

    def save_networks(self, filename_critic = 'neural-network-2-critic.pth', filename_actor = 'neural-network-2-actor.pth'):
        '''Save networks in working directory'''
        torch.save(self.main_critic_nn, filename_critic)
        torch.save(self.main_actor_nn, filename_actor)
        print(f'Saved main critic network as {filename_critic}')
        print(f'Saved main actor network as {filename_actor}')
        return


class CriticNeuralNetwork(nn.Module):
    ''' A feedforward neural network'''
    def __init__(self, input_size, actions_size):
        super().__init__()
        # params
        hidden_neurons_l1 = 400
        hidden_neurons_l2 = 200
        output_size = 1

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_neurons_l1)
        self.input_layer_activation = nn.ReLU()

        # Create hidden layers
        self.hidden = nn.Linear(hidden_neurons_l1 + actions_size, hidden_neurons_l2)   # Dim of previous neurons + actions
        self.hidden_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(hidden_neurons_l2, output_size)

    def forward(self, s, a):
        ''' Function used to compute the forward pass. Input: state (dim=8) and action (dim=2). Output: Q-value (dim=1)'''
        # Compute first layer
        l1 = self.input_layer(s)
        l1 = self.input_layer_activation(l1)

        # Compute hidden layers
        l2 = self.hidden(torch.cat([l1, a], dim=1))     # Concatenate output of l1 and action
        l2 = self.hidden_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)
        return out
    
    def save(self, filename='neural-network-2-critic.pth'):
        torch.save(self, filename)


class ActorNeuralNetwork(nn.Module):
    ''' A feedforward neural network'''
    def __init__(self, input_size, output_size):
        super().__init__()
        # params
        hidden_neurons_l1 = 400
        hidden_neurons_l2 = 200

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_neurons_l1)
        self.input_layer_activation = nn.ReLU()

        # Create hidden layers
        self.hidden = nn.Linear(hidden_neurons_l1, hidden_neurons_l2)
        self.hidden_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(hidden_neurons_l2, output_size)
        self.output_activation = nn.Tanh()

    def forward(self, s):
        ''' Function used to compute the forward pass. Input: state (dim=8). Output: action (dim=2)'''
        # Compute first layer
        l1 = self.input_layer(s)
        l1 = self.input_layer_activation(l1)

        # Compute hidden layers
        l2 = self.hidden(l1)
        l2 = self.hidden_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)
        out = self.output_activation(out)   # Apply tanh to constrain action between [-1, 1]
        return out

    def save(self, filename='neural-network-2-actor.pth'):
        torch.save(self, filename)
