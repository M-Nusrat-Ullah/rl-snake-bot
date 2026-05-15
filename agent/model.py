import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network — the brain of the agent.

    Architecture: 3 fully connected layers
      Input:  11 values (game state)
      Hidden: 256 neurons
      Output: 3 values (Q-value for each action)

    Q-value = "how good is this action in this state?"
    The agent picks the action with the highest Q-value.
    """

    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)