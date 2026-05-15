import os
import numpy as np
import matplotlib.pyplot as plt
from game.snake_env import SnakeEnv
from agent.agent import DQNAgent

# Training config
MAX_EPISODES        = 1000   # total episodes to train
TARGET_UPDATE_EVERY = 10     # sync target_net every N episodes
SAVE_EVERY          = 100    # save model checkpoint every N episodes
MODELS_DIR          = 'models'


def plot_training(scores, mean_scores, epsilons):
    """
    Plot training progress:
    - scores:       raw score per episode
    - mean_scores:  rolling average (smoothed trend)
    - epsilons:     exploration rate over time
    """
    plt.figure(figsize=(12, 5))

    # Score plot
    plt.subplot(1, 2, 1)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores, alpha=0.4, label='Score', color='blue')
    plt.plot(mean_scores, label='Mean Score', color='red', linewidth=2)
    plt.legend()

    # Epsilon plot
    plt.subplot(1, 2, 2)
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.plot(epsilons, color='green')

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()
    print("Plot saved to training_progress.png")


def train():
    # Setup
    os.makedirs(MODELS_DIR, exist_ok=True)
    env   = SnakeEnv(render_mode='none')   # no rendering during training (faster)
    agent = DQNAgent()

    scores      = []
    mean_scores = []
    epsilons    = []
    best_score  = 0
    total_score = 0

    print("=" * 50)
    print("  RL Snake Bot — Training Started")
    print("=" * 50)

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        # --- Run one full episode ---
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.store_experience(state, action, reward, next_state, terminated)
            loss = agent.learn()

            state = next_state
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        # --- After episode ---
        score = info['score']
        agent.decay_epsilon()

        # Sync target network
        if episode % TARGET_UPDATE_EVERY == 0:
            agent.update_target_network()

        # Save checkpoint
        if episode % SAVE_EVERY == 0:
            path = os.path.join(MODELS_DIR, f'model_ep{episode}.pth')
            agent.save(path)

        # Save best model
        if score > best_score:
            best_score = score
            agent.save(os.path.join(MODELS_DIR, 'best_model.pth'))

        # Tracking
        scores.append(score)
        total_score += score
        mean_score = total_score / episode
        mean_scores.append(mean_score)
        epsilons.append(agent.epsilon)

        # Log every 10 episodes
        if episode % 10 == 0:
            print(
                f"Episode {episode:>4} | "
                f"Score: {score:>3} | "
                f"Best: {best_score:>3} | "
                f"Mean: {mean_score:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Memory: {len(agent.memory):>6}"
            )

    # --- Training complete ---
    print("=" * 50)
    print(f"  Training Complete! Best Score: {best_score}")
    print("=" * 50)

    plot_training(scores, mean_scores, epsilons)
    env.close()


if __name__ == '__main__':
    train()