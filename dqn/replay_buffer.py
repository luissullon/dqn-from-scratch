"""
Fixed-size cyclic experience replay buffer.

Two problems replay solves for Q-learning with function approximation:
  1. Consecutive environment steps are highly correlated; SGD assumes
     roughly i.i.d. samples. Random minibatches break that correlation.
  2. Each transition can be reused many times instead of being thrown away
     after a single gradient step, which is far more sample-efficient.

Implemented as pre-allocated numpy arrays (not a Python deque of tuples) so
that sampling a batch is a handful of vectorized index operations rather
than a Python-level loop -- this matters once buffer_capacity reaches
10^5-10^6 transitions.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, state, action, reward, next_state, done) -> None:
        i = self._ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer with only {self._size} transitions."
            )
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
