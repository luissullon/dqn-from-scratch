"""
Function approximator for Q(s, a).

For low-dimensional state spaces (e.g. CartPole's 4-vector) a plain MLP is
sufficient and trains fast on CPU. The network outputs one Q-value per
discrete action in a single forward pass, which is what lets Q-learning's
max_a Q(s', a) be computed with one call instead of |A| calls.
"""

from typing import Sequence

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Multi-layer perceptron mapping state -> Q-values for every action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))

        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, obs_dim) -> Q-values: (batch, n_actions)."""
        return self.net(x)
