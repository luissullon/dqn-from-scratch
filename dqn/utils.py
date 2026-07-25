"""Small shared utilities: reproducibility and plotting helpers."""

import random
from typing import List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunningAverage:
    """Cheap O(1)-update moving average for logging (e.g. last-100-episode return)."""

    def __init__(self, window: int = 100):
        self.window = window
        self._values: List[float] = []

    def update(self, value: float) -> float:
        self._values.append(value)
        if len(self._values) > self.window:
            self._values.pop(0)
        return self.mean

    @property
    def mean(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0


def plot_training_curve(rewards: List[float], out_path: str, window: int = 20) -> None:
    """Save a PNG of episode reward + smoothed moving average. Import kept local
    so the rest of the package has no hard matplotlib dependency at import time."""
    import matplotlib.pyplot as plt

    rewards = np.array(rewards, dtype=np.float32)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid") if len(rewards) >= window else rewards

    plt.figure(figsize=(8, 5))
    plt.plot(rewards, alpha=0.3, label="Episode reward")
    if len(rewards) >= window:
        plt.plot(range(window - 1, len(rewards)), smoothed, label=f"{window}-episode moving average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
