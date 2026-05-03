"""
Stage 2 test: verify the network does what we think it does.

Things to check:
  1. Output shapes are right.
  2. Logits are roughly uniform at init (-> near-uniform action probs).
  3. Sampling works and gives valid actions.
  4. Gradients flow through both heads.
  5. log_prob and entropy have the right scale.
"""

import torch
import numpy as np
from network import ActorCritic
from torch.distributions import Categorical


def test_shapes():
    net = ActorCritic(obs_dim=2, n_actions=4)
    obs = torch.randn(8, 2)  # batch of 8
    logits, value = net(obs)
    assert logits.shape == (8, 4), f"logits shape wrong: {logits.shape}"
    assert value.shape  == (8,),    f"value shape wrong: {value.shape}"
    print("[PASS] forward shapes:", logits.shape, value.shape)


def test_init_is_near_uniform():
    """At init, with small actor-head weights, action probs should be ~uniform.
    If they're not, the agent commits to a single action before learning starts.
    """
    net = ActorCritic(obs_dim=2, n_actions=4)
    obs = torch.randn(100, 2)
    logits, _ = net(obs)
    probs = torch.softmax(logits, dim=-1)  # (100, 4)
    mean_probs = probs.mean(dim=0)
    # Each action should have prob roughly 0.25
    assert torch.allclose(mean_probs, torch.full((4,), 0.25), atol=0.02), \
        f"init policy not near-uniform: {mean_probs}"
    print(f"[PASS] init mean action probs: {mean_probs.tolist()} (expected ~0.25 each)")


def test_act():
    net = ActorCritic(obs_dim=2, n_actions=4)
    obs = torch.tensor([1.0, 0.0])  # single state, no batch dim
    obs = obs.unsqueeze(0)  # add batch dim -> (1, 2)
    a, logp, v, ent = net.act(obs)
    assert a in [0, 1, 2, 3]
    assert logp.shape == (1,)
    assert v.shape    == (1,)
    assert ent.shape  == (1,)
    # max entropy for 4 actions = log(4) ~ 1.386
    assert 0 < ent.item() <= np.log(4) + 1e-5
    print(f"[PASS] act() -> action={a}, log_prob={logp.item():.3f}, "
          f"value={v.item():.3f}, entropy={ent.item():.3f} (max={np.log(4):.3f})")


def test_gradients_flow():
    """Take a fake loss involving both heads and check that grads exist on
    every parameter. If a parameter has no grad, it's effectively dead."""
    net = ActorCritic(obs_dim=2, n_actions=4)
    obs = torch.randn(4, 2)
    logits, value = net(obs)
    # Fake combined loss (not the real A2C loss -- just checking plumbing)
    loss = logits.sum() + value.sum()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert p.grad.abs().sum() > 0, f"zero grad on {name}"
    print("[PASS] gradients flow through all parameters")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_shapes()
    test_init_is_near_uniform()
    test_act()
    test_gradients_flow()
    print("\nAll stage-2 tests passed.")
