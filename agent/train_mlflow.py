import os
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import torch

from game.snake_env import SnakeEnv
from agent.agent import DQNAgent
from config.training_config import CONFIG

MODELS_DIR = "models"


def plot_training(scores, mean_scores, epsilons, path="training_progress.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(scores, alpha=0.4, label="Score", color="blue")
    plt.plot(mean_scores, label="Mean Score", color="red", linewidth=2)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Exploration Rate (Epsilon)")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.plot(epsilons, color="green")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- MLflow setup ---
    mlflow.set_tracking_uri("mlruns")        # store locally in ./mlruns/
    mlflow.set_experiment("rl-snake-bot")    # experiment name

    with mlflow.start_run(run_name="dqn-baseline"):

        # Log all hyperparameters at once
        mlflow.log_params(CONFIG)

        env   = SnakeEnv(render_mode="none")
        agent = DQNAgent()

        scores      = []
        mean_scores = []
        epsilons    = []
        best_score  = 0
        total_score = 0

        print("=" * 50)
        print("  RL Snake Bot — Training with MLflow")
        print("=" * 50)

        for episode in range(1, CONFIG["max_episodes"] + 1):
            state, _ = env.reset()
            total_reward = 0

            while True:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.store_experience(state, action, reward, next_state, terminated)
                agent.learn()
                state = next_state
                total_reward += reward

                if terminated or truncated:
                    break

            score = info["score"]
            agent.decay_epsilon()

            if episode % CONFIG["target_update_every"] == 0:
                agent.update_target_network()

            # Save best model
            if score > best_score:
                best_score = score
                best_path  = os.path.join(MODELS_DIR, "best_model.pth")
                agent.save(best_path)

            # Save checkpoint
            if episode % CONFIG["save_every"] == 0:
                ckpt_path = os.path.join(MODELS_DIR, f"model_ep{episode}.pth")
                agent.save(ckpt_path)

            # Track metrics
            scores.append(score)
            total_score += score
            mean_score = total_score / episode
            mean_scores.append(mean_score)
            epsilons.append(agent.epsilon)

            # --- Log metrics to MLflow every episode ---
            mlflow.log_metrics({
                "score":      score,
                "mean_score": mean_score,
                "epsilon":    agent.epsilon,
                "best_score": best_score,
                "memory_size": len(agent.memory),
            }, step=episode)

            if episode % 10 == 0:
                print(
                    f"Episode {episode:>4} | "
                    f"Score: {score:>3} | "
                    f"Best: {best_score:>3} | "
                    f"Mean: {mean_score:.2f} | "
                    f"Epsilon: {agent.epsilon:.3f}"
                )

        # --- After training ---
        print("=" * 50)
        print(f"  Training Complete! Best Score: {best_score}")
        print("=" * 50)

        # Log final summary metrics
        mlflow.log_metrics({
            "final_best_score": best_score,
            "final_mean_score": mean_scores[-1],
        })

        # Log training plot as artifact
        plot_path = plot_training(scores, mean_scores, epsilons)
        mlflow.log_artifact(plot_path)

        # Log best model as artifact
        mlflow.log_artifact(os.path.join(MODELS_DIR, "best_model.pth"))

        # Log model properly with MLflow
        mlflow.pytorch.log_model(
            agent.policy_net,
            name="policy_net",
            serialization_format="pt2"
        )

        env.close()
        print("MLflow run complete. Launch UI with: mlflow ui")


if __name__ == "__main__":
    train()