# mlflow/config.py

CONFIG = {
    # Training
    "max_episodes":        1000,
    "target_update_every": 10,
    "save_every":          100,

    # Agent
    "batch_size":    1000,
    "gamma":         0.9,
    "lr":            0.001,
    "epsilon_start": 1.0,
    "epsilon_end":   0.01,
    "epsilon_decay": 0.995,

    # Memory
    "replay_capacity": 100_000,

    # Model
    "input_size":  11,
    "hidden_size": 256,
    "output_size": 3,

    # Environment
    "grid_width":  480,
    "grid_height": 480,
}