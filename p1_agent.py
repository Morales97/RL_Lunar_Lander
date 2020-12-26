# EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Support for Solution by Valentin Minoz, Daniel Morales

import numpy as np
import torch
from torch import nn
from torch import optim

class RandomAgent:
    ''' Agent taking actions uniformly at random'''
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Return uniformly random action (int) '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class QAgent:
    ''' Will take fully greedy actions from given neural net by filename
        Check yourself that action and state spaces match
    '''
    def __init__(self, neuralnetname):
        self.last_action = None
        self.net = torch.load(neuralnetname)

    def forward(self, state: np.ndarray):
        ''' Return ε-greedy action (int) '''
        q_values = self.net(torch.tensor([state]))
        _, action = torch.max(q_values, axis=1)
        self.last_action = action.item()
        return self.last_action


class DQNAgent:
    ''' Trainable Agent taking actions by DQN with ε-greedy exploration'''
    def __init__(self, n_actions, dim_state, discount, epsilon, learning_rate):
        self.n_actions = n_actions
        self.last_action = None

        # Params
        self.epsilon = epsilon
        self.disc = discount

        # Main Neural Network
        self.main_network = NeuralNetwork(dim_state, n_actions)

        # Targ Neural Network
        self.target_network = NeuralNetwork(dim_state, n_actions)

        # optimizer
        self.opt = optim.Adam(
            self.main_network.parameters(),
            lr=learning_rate)

    def forward(self, state: np.ndarray):
        ''' Return ε-greedy action (int) '''
        if np.random.random() < self.epsilon:
            self.last_action = np.random.randint(0, self.n_actions)
        else:
            q_values = self.main_network(torch.tensor([state]))
            _, action = torch.max(q_values, axis=1)
            self.last_action = action.item()
        return self.last_action

    def train(self, batch):
        '''One training step on agent'''
        states, actions, rewards, next_states, dones = zip(*batch)
        self.opt.zero_grad()

        # Compute target vals
        with torch.no_grad():
            actions = torch.tensor(actions, dtype = torch.int64)
            tar_q = self.target_network(torch.tensor(next_states))
            tar_maxq = tar_q.max(1)[0]
            # if input(tar_q): raise NameError
            # if input(tar_maxq): raise NameError
            tar_vals = (torch.tensor(rewards,dtype=torch.float32) +\
                (torch.tensor(dones)==False) * self.disc * tar_maxq)
            # if input(tar_vals): raise NameError

        # Compute output
        q_values = self.main_network.forward(torch.tensor(states,
                requires_grad=True,
                dtype = torch.float32)
                ).gather(1,actions.unsqueeze(1))
        q_values = q_values.reshape(-1) # collapse one dim

        # Update theta by performing backwards pass on mse-loss
        loss = nn.functional.mse_loss(q_values, tar_vals)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=2.)

        # Perform backward pass (backpropagation)
        self.opt.step()
        return

    def update_target(self):
        '''set target net to main net'''
        self.target_network.load_state_dict(
            self.main_network.state_dict())
        return

    def save_network(self, filename = 'neural-network-1.pth'):
        '''Save network in working directory'''
        torch.save(self.main_network, filename)
        print(f'Saved main_network as {filename}')
        return


class NeuralNetwork(nn.Module):
    ''' A feedforward neural network'''
    def __init__(self, input_size, output_size):
        super().__init__()
        # params
        l1out = 64 #32 turns out just as good
        l2out = 64

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, l1out)
        self.input_layer_activation = nn.ReLU()

        # Create hidden layers
        self.hidden = nn.Linear(l1out, l2out)
        self.hidden_a = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(l2out, output_size)

    def forward(self, x):
        ''' Function used to compute the forward pass'''
        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Compute hidden layers
        l2 = self.hidden(l1)
        l2 = self.hidden_a(l2)

        # Compute output layer
        out = self.output_layer(l2)
        return out
