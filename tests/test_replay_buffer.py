import numpy as np
import pytest

from dqn.replay_buffer import ReplayBuffer


def make_buffer(capacity=10, obs_dim=4, seed=0):
    return ReplayBuffer(capacity=capacity, obs_dim=obs_dim, seed=seed)


def test_starts_empty():
    buf = make_buffer()
    assert len(buf) == 0


def test_push_increases_length():
    buf = make_buffer(capacity=5)
    state = np.zeros(4, dtype=np.float32)
    for i in range(3):
        buf.push(state, 0, 1.0, state, False)
    assert len(buf) == 3


def test_length_caps_at_capacity():
    buf = make_buffer(capacity=5)
    state = np.zeros(4, dtype=np.float32)
    for i in range(20):
        buf.push(state, 0, 1.0, state, False)
    assert len(buf) == 5


def test_cyclic_overwrite():
    """After wrapping around, the buffer should contain only the most recent `capacity` items."""
    buf = make_buffer(capacity=3, obs_dim=1)
    for i in range(5):
        state = np.array([i], dtype=np.float32)
        buf.push(state, i, float(i), state, False)

    # Only the last 3 pushes (indices 2, 3, 4) should be present, in some order.
    present_actions = sorted(buf.actions[: len(buf)].tolist())
    assert present_actions == [2, 3, 4]


def test_sample_shapes():
    buf = make_buffer(capacity=50, obs_dim=4)
    state = np.zeros(4, dtype=np.float32)
    for i in range(20):
        buf.push(state, i % 2, 1.0, state, i % 7 == 0)

    states, actions, rewards, next_states, dones = buf.sample(8)
    assert states.shape == (8, 4)
    assert next_states.shape == (8, 4)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)


def test_sample_raises_if_not_enough_data():
    buf = make_buffer(capacity=50, obs_dim=4)
    state = np.zeros(4, dtype=np.float32)
    buf.push(state, 0, 1.0, state, False)

    with pytest.raises(ValueError):
        buf.sample(8)


def test_sample_is_reproducible_with_seed():
    buf1 = make_buffer(capacity=50, obs_dim=4, seed=42)
    buf2 = make_buffer(capacity=50, obs_dim=4, seed=42)
    state = np.zeros(4, dtype=np.float32)
    for buf in (buf1, buf2):
        for i in range(30):
            buf.push(np.array([i, i, i, i], dtype=np.float32), i, float(i), state, False)

    s1, a1, r1, ns1, d1 = buf1.sample(10)
    s2, a2, r2, ns2, d2 = buf2.sample(10)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(r1, r2)
