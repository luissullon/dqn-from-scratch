import numpy as np
import torch

from dqn import Config, DQNAgent


def make_agent(**overrides):
    cfg = Config(
        env_id="CartPole-v1",
        hidden_sizes=(16, 16),
        batch_size=8,
        min_replay_size=8,
        buffer_capacity=200,
        device="cpu",
        **overrides,
    )
    return DQNAgent(obs_dim=4, n_actions=2, cfg=cfg)


def fill_buffer(agent, n=20):
    for _ in range(n):
        s = np.random.randn(4).astype(np.float32)
        ns = np.random.randn(4).astype(np.float32)
        agent.remember(s, np.random.randint(2), float(np.random.randn()), ns, False)


def test_act_returns_valid_action():
    agent = make_agent()
    state = np.random.randn(4).astype(np.float32)
    action = agent.act(state, step=0)
    assert action in (0, 1)


def test_greedy_action_ignores_epsilon():
    agent = make_agent(eps_start=1.0, eps_end=1.0)
    state = np.random.randn(4).astype(np.float32)
    # With eps=1 exploration would be random; greedy=True should bypass it
    # and always defer to argmax Q, which is deterministic given fixed weights.
    a1 = agent.act(state, step=0, greedy=True)
    a2 = agent.act(state, step=0, greedy=True)
    assert a1 == a2


def test_ready_to_learn_flag():
    agent = make_agent()
    assert not agent.ready_to_learn()
    fill_buffer(agent, n=agent.cfg.min_replay_size)
    assert agent.ready_to_learn()


def test_learn_reduces_or_returns_finite_loss():
    agent = make_agent()
    fill_buffer(agent, n=50)
    loss = agent.learn()
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_target_network_updates_periodically():
    agent = make_agent(target_update_interval=3)
    fill_buffer(agent, n=50)

    before = [p.clone() for p in agent.target_net.parameters()]
    for _ in range(3):
        agent.learn()
    after = list(agent.target_net.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed
