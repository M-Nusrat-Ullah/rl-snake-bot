import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game.snake_game import SnakeGame


class SnakeEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around SnakeGame.

    observation_space: 11-dimensional binary vector (from get_state())
    action_space:      3 discrete actions (straight, right, left)
    """

    metadata = {'render_modes': ['human', 'none'], 'render_fps': 40}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
        self.game = SnakeGame(render=(render_mode == 'human'))

        # State: 11 binary values
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(11,),
            dtype=np.int32
        )

        # Actions: 0=straight, 1=right turn, 2=left turn
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs = self.game.get_state()
        info = {}
        return obs, info

    def step(self, action):
        # Convert int action to one-hot vector
        action_one_hot = [0, 0, 0]
        action_one_hot[action] = 1

        reward, done, score = self.game.play_step(action_one_hot)
        obs = self.game.get_state()

        # Gymnasium requires: obs, reward, terminated, truncated, info
        terminated = done
        truncated  = False
        info = {'score': score}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # Pygame renders inside play_step() already

    def close(self):
        import pygame
        pygame.quit()