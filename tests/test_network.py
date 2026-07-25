import torch

from dqn.network import QNetwork


def test_output_shape():
    net = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=(32, 32))
    x = torch.randn(8, 4)
    out = net(x)
    assert out.shape == (8, 2)


def test_single_sample_forward():
    net = QNetwork(obs_dim=6, n_actions=3, hidden_sizes=(16,))
    x = torch.randn(1, 6)
    out = net(x)
    assert out.shape == (1, 3)


def test_gradients_flow():
    net = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=(32,))
    x = torch.randn(4, 4)
    out = net(x)
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


def test_deterministic_given_same_weights():
    net = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=(16,))
    net.eval()
    x = torch.randn(2, 4)
    out1 = net(x)
    out2 = net(x)
    torch.testing.assert_close(out1, out2)
