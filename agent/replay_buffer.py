import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer — the agent's memory.

    Instead of learning from each experience immediately (which is unstable),
    we store experiences and sample random batches to learn from.

    Why random sampling?
    - Breaks correlation between consecutive experiences
    - Same experience can be learned from multiple times
    - Makes training much more stable

    Each experience is a tuple of:
        (state, action, reward, next_state, done)
    """

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
        # deque automatically removes oldest entries when full

    def push(self, state, action, reward, next_state, done):
        """Store one experience in memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.
        Returns separate numpy arrays for each component.
        """
        batch = random.sample(self.buffer, batch_size)

        states      = np.array([e[0] for e in batch], dtype=np.float32)
        actions     = np.array([e[1] for e in batch], dtype=np.int64)
        rewards     = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones       = np.array([e[4] for e in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)