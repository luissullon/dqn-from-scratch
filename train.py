"""
Train a DQN agent.

Usage
-----
    python train.py --env CartPole-v1 --steps 150000
    python train.py --config configs/cartpole.yaml

The loop follows the classic DQN structure (Mnih et al., 2015):
  1. Act epsilon-greedily in the environment, store the transition.
  2. Once the buffer has enough data, sample a minibatch and take one
     gradient step toward the Bellman target (see dqn/agent.py + README.md).
  3. Periodically hard-update the target network and checkpoint to disk.
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import yaml

from dqn import Config, DQNAgent
from dqn.utils import set_seed, RunningAverage, plot_training_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DQN agent.")
    p.add_argument("--config", type=str, default=None, help="Path to a yaml config file.")
    p.add_argument("--env", type=str, default=None, help="Override env_id.")
    p.add_argument("--steps", type=int, default=None, help="Override total_steps.")
    p.add_argument("--seed", type=int, default=None, help="Override seed.")
    p.add_argument("--no-double-dqn", action="store_true", help="Disable Double DQN target.")
    p.add_argument("--run-name", type=str, default=None, help="Subdirectory under checkpoint_dir.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    if args.config:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        cfg = Config(**raw)
    else:
        cfg = Config()

    if args.env:
        cfg.env_id = args.env
    if args.steps:
        cfg.total_steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_double_dqn:
        cfg.use_double_dqn = False

    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    run_name = args.run_name or f"{cfg.env_id}_{int(time.time())}"
    run_dir = os.path.join(cfg.checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))

    env = gym.make(cfg.env_id)
    env.action_space.seed(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, cfg)

    episode_rewards = []
    avg_reward = RunningAverage(window=100)

    state, _ = env.reset(seed=cfg.seed)
    episode_reward, episode_len, episode_idx = 0.0, 0, 0

    print(f"Training on {cfg.env_id} | device={agent.device} | double_dqn={cfg.use_double_dqn}")
    t0 = time.time()

    for step in range(1, cfg.total_steps + 1):
        action = agent.act(state, step)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.remember(state, action, reward, next_state, terminated)
        state = next_state
        episode_reward += reward
        episode_len += 1

        if agent.ready_to_learn() and step % cfg.train_frequency == 0:
            agent.learn()

        if done or episode_len >= cfg.max_episode_steps:
            episode_rewards.append(episode_reward)
            running_mean = avg_reward.update(episode_reward)
            episode_idx += 1

            if episode_idx % cfg.log_interval == 0:
                elapsed = time.time() - t0
                eps = agent.epsilon(step)
                print(
                    f"step={step:>7} | episode={episode_idx:>5} | "
                    f"reward={episode_reward:>6.1f} | avg100={running_mean:>6.1f} | "
                    f"eps={eps:.3f} | elapsed={elapsed:6.1f}s"
                )

            state, _ = env.reset()
            episode_reward, episode_len = 0.0, 0

        if step % cfg.checkpoint_interval == 0:
            agent.save(os.path.join(run_dir, f"checkpoint_{step}.pt"))

    agent.save(os.path.join(run_dir, "checkpoint_final.pt"))
    np.save(os.path.join(run_dir, "episode_rewards.npy"), np.array(episode_rewards))
    try:
        plot_training_curve(episode_rewards, os.path.join(run_dir, "training_curve.png"))
    except ImportError:
        pass  # matplotlib not installed; skip plotting

    env.close()
    print(f"Done. Artifacts saved to {run_dir}/")


if __name__ == "__main__":
    main()
