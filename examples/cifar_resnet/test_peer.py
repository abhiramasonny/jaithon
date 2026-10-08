"""Checks the torch peer's evaluation on the CPU, in a second or two:

    ~/.venvs/scratch/bin/python examples/cifar_resnet/test_peer.py

The batch-statistics evaluation runs BatchNorm in training mode, which also
updates the running mean and variance. It must put them back, or the
running-statistics number printed after it is partly computed from the test
set.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True  # no __pycache__ beside the example
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar_resnet as peer  # noqa: E402

peer.dev = torch.device("cpu")


def trained_net():
    torch.manual_seed(0)
    net = peer.ResNet(8, 4)
    peer.init_like_jaitensor(net)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    net.train()
    for _ in range(4):
        x = torch.randn(32, 3, 32, 32) * 0.5 + 0.2
        y = torch.randint(0, 10, (32,))
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def test_batch_statistics_leave_the_buffers_alone():
    net = trained_net()
    x = torch.randn(2000, 3, 32, 32) * 0.7 - 0.1
    y = torch.randint(0, 10, (2000,))
    before = [b.clone() for b in net.buffers()]
    peer.evaluate(net, x, y, True)
    after = list(net.buffers())
    assert len(before) == len(after) > 0
    for b, a in zip(before, after):
        assert torch.equal(b, a), "evaluate(batch_stats=True) changed a BatchNorm buffer"


def test_running_statistics_do_not_depend_on_the_order():
    net = trained_net()
    x = torch.randn(2000, 3, 32, 32) * 0.7 - 0.1
    y = torch.randint(0, 10, (2000,))
    first = peer.evaluate(net, x, y, False)
    logits_first = net.eval()(x[:64]).detach().clone()
    peer.evaluate(net, x, y, True)
    second = peer.evaluate(net, x, y, False)
    logits_second = net.eval()(x[:64]).detach()
    assert first == second
    assert torch.equal(logits_first, logits_second)


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
