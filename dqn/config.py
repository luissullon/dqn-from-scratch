"""
Central place for every hyperparameter the agent / training loop uses.

Keeping this as a single dataclass (instead of scattering constants across
files or hiding them in argparse) makes runs reproducible: a `Config` object
can be logged, saved to yaml, and re-loaded to reconstruct an experiment
exactly.
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    # --- Environment -----------------------------------------------------
    env_id: str = "CartPole-v1"
    seed: int = 0

    # --- Network -----------------------------------------------------------
    hidden_sizes: tuple = (128, 128)

    # --- Optimization -------------------------------------------------------
    lr: float = 2.5e-4
    gamma: float = 0.99          # discount factor
    batch_size: int = 64
    grad_clip_norm: float = 10.0

    # --- Replay buffer -------------------------------------------------------
    buffer_capacity: int = 100_000
    min_replay_size: int = 1_000  # warm-up steps before learning starts

    # --- Exploration (epsilon-greedy, linear decay) --------------------------
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 25_000

    # --- Target network --------------------------------------------------
    target_update_interval: int = 1_000  # steps, hard update
    use_double_dqn: bool = True          # Double DQN target (van Hasselt et al., 2016)

    # --- Training loop -----------------------------------------------------
    total_steps: int = 150_000
    train_frequency: int = 1        # env steps between gradient updates
    max_episode_steps: int = 500
    log_interval: int = 10          # episodes

    # --- Misc --------------------------------------------------------------
    device: str = "auto"            # "auto" | "cpu" | "cuda"
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 25_000

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
        return cls(**data)
