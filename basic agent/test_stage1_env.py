"""
Stage 1 test: run a random policy and check basic env behavior.

Things we want to verify:
  1. reset() returns a 2D observation in [0, 1].
  2. step() returns (obs, reward, done, truncated, info).
  3. Walls work (can't go off the grid).
  4. The goal is reachable -> a random policy occasionally finishes.
  5. max_steps truncation works.
"""

import numpy as np
from gridworld import GridWorld


def test_basic_api():
    env = GridWorld()
    obs = env.reset()
    assert obs.shape == (2,), f"obs should be (2,), got {obs.shape}"
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    print("[PASS] reset() returns valid observation:", obs)

    obs, r, done, trunc, info = env.step(0)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    print("[PASS] step() returns 5-tuple")


def test_walls():
    env = GridWorld()
    env.reset()  # agent at (4, 0)
    # Try to go LEFT off the grid; position should not change.
    pos_before = env.pos.copy()
    env.step(3)  # left
    assert np.array_equal(env.pos, pos_before), "agent walked through left wall"
    # Try to go DOWN off the grid.
    env.step(2)  # down
    assert np.array_equal(env.pos, pos_before), "agent walked through bottom wall"
    print("[PASS] walls block movement")


def test_random_policy(n_episodes=200, seed=0):
    rng = np.random.default_rng(seed)
    env = GridWorld()
    successes, lengths = 0, []
    for ep in range(n_episodes):
        env.reset()
        done = trunc = False
        steps = 0
        while not (done or trunc):
            a = int(rng.integers(0, 4))
            _, r, done, trunc, _ = env.step(a)
            steps += 1
        if done:
            successes += 1
            lengths.append(steps)
    rate = successes / n_episodes
    print(f"[INFO] random policy success rate: {rate:.1%} "
          f"({successes}/{n_episodes}), "
          f"mean ep length when solved: {np.mean(lengths) if lengths else float('nan'):.1f}")
    # Random policy on 5x5 with 50-step cap should solve some episodes by chance.
    # Min path length is 8 (4 ups + 4 rights). With 50 steps, success rate
    # will not be zero -- somewhere around 20-50%.
    assert rate > 0.05, "random policy never reaches goal -- env is broken"
    print("[PASS] goal is reachable")


if __name__ == "__main__":
    test_basic_api()
    test_walls()
    test_random_policy()
    print("\nAll stage-1 tests passed.")
