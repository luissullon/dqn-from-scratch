"""
DQNAgent: action selection, the Bellman-target computation, and the
gradient step. This is the piece of the codebase that most directly
implements the math described in README.md -- see that file for the full
derivation of the loss used in `learn()` below.
"""

import copy
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dqn.config import Config
from dqn.network import QNetwork
from dqn.replay_buffer import ReplayBuffer


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: Config):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = _resolve_device(cfg.device)

        self.q_net = QNetwork(obs_dim, n_actions, cfg.hidden_sizes).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_capacity, obs_dim, seed=cfg.seed)

        self._train_steps = 0

    # ------------------------------------------------------------------ #
    # Exploration
    # ------------------------------------------------------------------ #
    def epsilon(self, step: int) -> float:
        """Linear decay from eps_start to eps_end over eps_decay_steps."""
        frac = min(1.0, step / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, state: np.ndarray, step: int, greedy: bool = False) -> int:
        """epsilon-greedy action selection w.r.t. the online network."""
        eps = 0.0 if greedy else self.epsilon(step)
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def ready_to_learn(self) -> bool:
        return len(self.buffer) >= max(self.cfg.min_replay_size, self.cfg.batch_size)

    def learn(self) -> float:
        """
        One gradient step on the Bellman-error loss.

        Standard DQN target:      y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        Double DQN target:        y = r + gamma * (1 - done) * Q_target(s', argmax_a' Q_online(s', a'))

        Double DQN decouples action *selection* (online net) from action
        *evaluation* (target net) to reduce the maximization bias that
        plain Q-learning's max operator introduces -- see README section
        "Overestimation bias & Double DQN".
        """
        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)

        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)

        # Q(s, a) for the actions actually taken.
        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.cfg.use_double_dqn:
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values

            targets = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_values, targets)  # Huber loss: robust to outlier TD errors

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.cfg.target_update_interval == 0:
            self.update_target()

        return float(loss.item())

    def update_target(self) -> None:
        """Hard update: copy online weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self._train_steps,
                "config": self.cfg.to_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._train_steps = ckpt["train_steps"]
