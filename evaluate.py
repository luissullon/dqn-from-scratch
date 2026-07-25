"""
Run a trained agent greedily (epsilon=0) and report average return.

Usage
-----
    python evaluate.py --checkpoint checkpoints/CartPole-v1_123/checkpoint_final.pt --episodes 20
    python evaluate.py --checkpoint <path> --render
"""

import argparse

import gymnasium as gym
import numpy as np

from dqn import Config, DQNAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = Config(**ckpt["config"])

    env = gym.make(cfg.env_id, render_mode="human" if args.render else None)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, cfg)
    agent.load(args.checkpoint)

    returns = []
    for ep in range(args.episodes):
        state, _ = env.reset(seed=1000 + ep)
        done, ep_return = False, 0.0
        while not done:
            action = agent.act(state, step=0, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
        print(f"Episode {ep + 1:>3}: return = {ep_return:.1f}")

    print(f"\nMean return over {args.episodes} episodes: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    env.close()


if __name__ == "__main__":
    main()
