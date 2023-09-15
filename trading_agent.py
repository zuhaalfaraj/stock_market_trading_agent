import torch
import torch.nn as nn
from model import QNetwork
from collections import deque
import random
import numpy as np

class DQAgent:
    """
    Deep Q-Network (DQN) agent for reinforcement learning.

    Args:
        input_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the action space.
        hidden_dim (int, optional): Dimension of the hidden layers in the Q-network. Defaults to 128.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        epsilon (float, optional): Exploration rate for epsilon-greedy policy. Defaults to 0.5.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.000001.

    Attributes:
        device (str): Device used for training ('cuda' or 'cpu').
        q_net (QNetwork): Q-network for estimating Q-values.
        target_net (QNetwork): Target Q-network for stable training.
        optimizer (torch.optim): Optimizer for training the Q-network.
        loss_fn (nn.MSELoss): Loss function for Q-value estimation.
        memory (deque): Replay memory for experience storage.
        batch_size (int): Size of the mini-batch for replay.
        epsilon (float): Current exploration rate.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Exploration rate decay factor.
        loss_history (list): History of training losses.
    """

    def __init__(self, input_dim, action_dim, hidden_dim=128, gamma=0.99, epsilon=0.9,epsilon_min = 0.01,
                 epsilon_decay = 0.999, lr=0.0000001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(input_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay =epsilon_decay

        self.loss_history = []
        self.loss_history_avg = []

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience tuple in the replay memory.

        Args:
            state (torch.Tensor): Current state.
            action (int): Chosen action.
            reward (float): Received reward.
            next_state (torch.Tensor): Next state.
            done (bool): Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self, filename="best_model.pth"):
        """
        Save the Q-network model to a specified file.

        Args:
            filename (str): Name of the saved model file.
        """
        torch.save(self.q_net.state_dict(), 'models/' + filename)

    def load_model(self, path):
        """
        Load model from a specific directory.

        Args:
            path (str): Path to the saved model.
        """
        self.q_net.load_state_dict(torch.load(path))
        self.q_net.eval()

    def act(self, state, available_actions):
        """
        Choose an action based on the current state and available actions.

        Args:
            state (torch.Tensor): Current state.
            available_actions (list): List of available actions.

        Returns:
            int: Chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)

        q_values = self.q_net(state)
        sorted_actions = torch.argsort(q_values, descending=True).cpu().detach().numpy()[0]

        for action in sorted_actions:
            if action in available_actions:
                return action

        raise ValueError("No valid actions provided!")

    def replay(self):
        """Perform a replay step to update the Q-network."""
        if len(self.memory) < self.batch_size:
            return

        start_idx = random.randint(0, len(self.memory) - self.batch_size)
        batch = [self.memory[i] for i in range(start_idx, start_idx + self.batch_size)]

        state, action, reward, next_state, done = zip(*batch)

        state = torch.stack(state).squeeze(1).to(self.device)
        next_state = torch.stack(next_state).squeeze(1).to(self.device)
        action = torch.tensor(action).unsqueeze(1).to(self.device)
        reward = torch.Tensor(reward).unsqueeze(1).to(self.device)
        done = torch.Tensor(done).unsqueeze(1).to(self.device)

        current_q_values = self.q_net(state).gather(1, action)

        next_q_values = self.target_net(next_state).max(dim=1)[0]

        target_q_values = reward + (self.gamma * next_q_values.unsqueeze(1) * (1 - done))

        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q_values, target_q_values)

        self.loss_history.append(loss.item())
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        """Update the target Q-network with the current Q-network's weights."""
        self.target_net.load_state_dict(self.q_net.state_dict())
