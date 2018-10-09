from abc import abstractmethod

import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from model import DuelingQNetwork
from model import ConvQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 0.0075  # for soft update of target parameters
LR = 0.001  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, buffer_size, batch_size, gamma, tau, lr, update_rate, layers,
                 network_type, memory=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_rate = update_rate

        # Q-Network
        if network_type == 'Simple':
            self.qnetwork_local = QNetwork(state_size, action_size, seed, layers).to(device)

            self.qnetwork_target = QNetwork(state_size, action_size, seed, layers).to(device)
        elif network_type == 'Dueling':
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed, layers).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed, layers).to(device)
        elif network_type == 'SimpleConvolution':
            self.qnetwork_local = ConvQNetwork().to(device)
            self.qnetwork_target = ConvQNetwork().to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        print(self.qnetwork_local)
        # Replay memory
        if memory is None:
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        else:
            self.memory = memory
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def set_lr(self, lr):
        self.lr = lr
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    @abstractmethod
    def loss(self, actions, state, next_state, rewards, dones):
        pass

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        loss = self.loss(actions, states, next_states, rewards, dones)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # self.experience =
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DQNAgent(Agent):
    def loss(self, actions, states, next_states, rewards, dones):
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[
            0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss


class DDQNAgent(Agent):
    def loss(self, actions, states, next_states, rewards, dones):
        # Get best actions from local network
        Q_max = self.qnetwork_local(next_states).detach().max(1)[
            1].unsqueeze(1)
        # Get estimated Q values for selected actions from target network
        Q_next = self.qnetwork_target(next_states).gather(1, Q_max)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute the loss.
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss
