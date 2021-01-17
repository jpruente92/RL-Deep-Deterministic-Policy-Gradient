import numpy as np
import random
import copy

from networks import Policy_network, Q_network

import torch
import torch.nn.functional as F
import torch.optim as optim

from hyperparameters import *


class Agent():

    def __init__(self, state_size, action_size, random_seed,device):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.device=device

        # Policy/Actor Network (w/ Target Network)
        self.policy_network_local = Policy_network(state_size, action_size, random_seed,"Neural_networks/"+FILENAME_FOR_LOADING+"policy_local.pth").to(device)
        self.policy_network_target = Policy_network(state_size, action_size, random_seed,"Neural_networks/"+FILENAME_FOR_LOADING+"policy_target.pth").to(device)
        self.policy_network_optimizer = optim.Adam(self.policy_network_local.parameters(), lr=LR_ACTOR)

        # Qvalue/Critic Network (w/ Target Network)
        self.qvalue_network_local = Q_network(state_size, action_size, random_seed,"Neural_networks/"+FILENAME_FOR_LOADING+"qvalue_local.pth").to(device)
        self.qvalue_network_target = Q_network(state_size, action_size, random_seed,"Neural_networks/"+FILENAME_FOR_LOADING+"qvalue_target.pth").to(device)
        self.qvalue_network_optimizer = optim.Adam(self.qvalue_network_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)


        # Noise process
        self.noise = OUNoise(action_size, random_seed)



    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action=self.policy_network_local.evaluate(state,False).cpu().data.numpy()
        if TRAINMODE:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, gamma,replay_buffer):
        experiences = replay_buffer.sample()
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update qvalue network with td error ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.policy_network_target(next_states)
        Q_next_state = self.qvalue_network_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_next_state * (1 - dones))
        # Compute qvalue loss
        Q_current_state = self.qvalue_network_local(states, actions)
        qvalue_loss = F.mse_loss(Q_current_state, Q_targets)
        # Minimize the loss
        self.qvalue_network_optimizer.zero_grad()
        qvalue_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qvalue_network_local.parameters(), 1)
        self.qvalue_network_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.policy_network_local(states)
        policy_loss = -self.qvalue_network_local(states, actions_pred).mean()
        # Minimize the loss
        self.policy_network_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_network_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.qvalue_network_local, self.qvalue_network_target, TAU)
        self.soft_update(self.policy_network_local, self.policy_network_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_networks(self):
        self.policy_network_local.save("Neural_networks/"+FILENAME_FOR_SAVING+"policy_local.pth")
        self.policy_network_target.save("Neural_networks/"+FILENAME_FOR_SAVING+"policy_target.pth")
        self.qvalue_network_local.save("Neural_networks/"+FILENAME_FOR_SAVING+"qvalue_local.pth")
        self.qvalue_network_target.save("Neural_networks/"+FILENAME_FOR_SAVING+"qvalue_target.pth")

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.rand() for i in range(len(x))])
        self.state = x + dx
        return self.state


