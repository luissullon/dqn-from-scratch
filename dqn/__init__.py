"""
dqn: a small, modular, from-scratch implementation of Deep Q-Networks.

Public API
----------
QNetwork        -- the Q-value function approximator (MLP or CNN)
ReplayBuffer     -- fixed-size cyclic experience replay buffer
DQNAgent         -- ties network + buffer + target network into an agent
Config           -- dataclass of hyperparameters
"""

from dqn.network import QNetwork
from dqn.replay_buffer import ReplayBuffer, Transition
from dqn.agent import DQNAgent
from dqn.config import Config

__all__ = ["QNetwork", "ReplayBuffer", "Transition", "DQNAgent", "Config"]
__version__ = "1.0.0"
