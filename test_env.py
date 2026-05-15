"""
Manual test: Run the environment with RANDOM actions.
You should see the Snake window open and the snake moving randomly.
This confirms our environment works before we attach any AI to it.
"""
from game.snake_env import SnakeEnv
import time

env = SnakeEnv(render_mode='human')
obs, info = env.reset()

print(f"Observation space: {env.observation_space}")
print(f"Action space:      {env.action_space}")
print(f"Initial state:     {obs}")

for episode in range(3):
    obs, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            print(f"Episode {episode+1} | Steps: {steps} | Score: {info['score']} | Total Reward: {total_reward}")
            break

env.close()