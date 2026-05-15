import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from agent.model import DQN
from agent.replay_buffer import ReplayBuffer

# Hyperparameters
BATCH_SIZE    = 1000    # how many experiences to learn from at once
GAMMA         = 0.9     # discount factor (how much to value future rewards)
LR            = 0.001   # learning rate
EPSILON_START = 1.0     # start fully random (100% exploration)
EPSILON_END   = 0.01    # minimum randomness (1%)
EPSILON_DECAY = 0.995   # how fast randomness decreases each episode


class DQNAgent:
    """
    The DQN Agent — combines everything together.

    It has two networks:
    - policy_net:  the network being trained (makes decisions)
    - target_net:  a frozen copy, updated slowly (stable learning target)

    Why two networks?
    If we train against a moving target, learning is unstable.
    The target_net gives us a stable Q-value target to aim for.
    Every N episodes we copy policy_net → target_net.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Two networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target_net is never trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory    = ReplayBuffer()
        self.epsilon   = EPSILON_START
        self.steps     = 0

    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon  → random action (exploration)
        With probability 1-epsilon → best action from network (exploitation)

        Early in training: mostly random (explore the game)
        Later in training: mostly network (exploit what it learned)
        """
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # random action

        # Convert state to tensor and get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()  # action with highest Q-value

    def store_experience(self, state, action, reward, next_state, done):
        """Save experience to replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Sample a batch from memory and train the policy network.

        The Q-learning update rule:
        Q(s,a) = reward + gamma * max(Q(s', a')) * (1 - done)

        We compute:
        - current_q:  what the policy_net currently thinks Q(s,a) is
        - target_q:   what it SHOULD be (using target_net for stability)
        - loss:       how far off we are (MSE)
        - backprop:   nudge the network to reduce the loss
        """
        if len(self.memory) < BATCH_SIZE:
            return  # not enough experiences yet, skip training

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Convert to tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values from policy_net
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values from target_net (no gradient needed)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        # Compute loss and backpropagate
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """Reduce epsilon after each episode — less exploration over time."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        """Copy policy_net weights → target_net (done every N episodes)."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path='model.pth'):
        """Save trained model to disk."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path='model.pth'):
        """Load a trained model from disk."""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")