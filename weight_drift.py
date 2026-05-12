"""
Frobenius-norm comparison of weights: trained network vs an untrained baseline.

For each layer of the network, compute:
    || W_trained - W_baseline ||_F

This measures the total magnitude of weight change due to training (assuming
the baseline shares the same initialization scheme). Larger = more learning.

We also report a control: || W_baseline_1 - W_baseline_2 ||_F, the typical
distance between two random initializations with different seeds. The
ratio (trained-vs-baseline) / (baseline-vs-baseline) tells you how much more
the trained weights moved than chance.

Usage:
    python weight_drift.py
"""

import torch
import numpy as np

from network import ActorCritic
from two_cue_gridworld import OBS_DIM


def frobenius_diff(net_a, net_b):
    """For each named parameter, compute ||A - B||_F. Returns a dict of layer -> norm."""
    diffs = {}
    state_a = net_a.state_dict()
    state_b = net_b.state_dict()
    # Both nets should have the same parameter names since they're the
    # same architecture. If not, that's a bug we want to know about.
    assert state_a.keys() == state_b.keys(), \
        "networks have different parameter names -- architecture mismatch?"
    for name in state_a:
        # .norm() with no args is the Frobenius norm for 2D tensors,
        # and the L2 (Euclidean) norm for 1D tensors. Same idea: sqrt(sum of squares).
        diff = (state_a[name] - state_b[name]).norm().item()
        diffs[name] = diff
    return diffs


def report_drift(trained_path="trained_agent.pt", hidden=400,
                 baseline_seed_1=999, baseline_seed_2=1000):
    """Compare trained network to a fresh untrained one, plus a chance baseline."""

    # 1. Load the trained net.
    trained = ActorCritic(obs_dim=OBS_DIM, n_actions=4, hidden=hidden)
    trained.load_state_dict(torch.load(trained_path))

    # 2. Make two fresh, differently-seeded untrained nets.
    #    - One is the "baseline" we compare the trained net to.
    #    - The other is for the random-vs-random control.
    torch.manual_seed(baseline_seed_1)
    baseline_1 = ActorCritic(obs_dim=OBS_DIM, n_actions=4, hidden=hidden)

    torch.manual_seed(baseline_seed_2)
    baseline_2 = ActorCritic(obs_dim=OBS_DIM, n_actions=4, hidden=hidden)

    # 3. Compute Frobenius differences.
    trained_vs_baseline   = frobenius_diff(trained,    baseline_1)
    baseline_vs_baseline  = frobenius_diff(baseline_1, baseline_2)

    # 4. Print a table.
    print(f"{'layer':<30} {'trained vs init':>15} {'random vs random':>18} {'ratio':>14}")
    print("-" * 82)
    for name in trained_vs_baseline:
        a = trained_vs_baseline[name]
        b = baseline_vs_baseline[name]
        # Skip the ratio if the chance baseline is essentially zero -- it's
        # uninformative (we'd be dividing by a number that's small for
        # initialization-design reasons, not because the layer is "interesting").
        if b < 1e-3:
            ratio_str = "n/a (init~0)"
        else:
            ratio_str = f"{a/b:.2f}x"
        print(f"{name:<30} {a:>15.3f} {b:>18.3f} {ratio_str:>14}")

    # 5. Highlight the layer asked about specifically:
    # "weights going TO the last hidden layer" = trunk.2.weight
    target_layer = "trunk.2.weight"
    if target_layer in trained_vs_baseline:
        a = trained_vs_baseline[target_layer]
        b = baseline_vs_baseline[target_layer]
        print()
        print(f"** Weights going to the last hidden layer ({target_layer}):")
        print(f"   ||W_trained - W_untrained||_F = {a:.3f}")
        print(f"   ||W_rand1   - W_rand2||_F     = {b:.3f}  (chance baseline)")
        print(f"   ratio                         = {a/b:.2f}x")
        if a / b > 1.5:
            print(f"   -> trained weights moved noticeably more than chance.")
        else:
            print(f"   -> weights barely moved beyond chance. Did training do much here?")

    return trained_vs_baseline, baseline_vs_baseline


if __name__ == "__main__":
    report_drift()
