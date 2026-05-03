"""
Sanity checks for the two-cue task.

We're verifying:
  1. Observation shape and dtype.
  2. Cues are sampled fresh each episode.
  3. Odor stays hidden until the LIGHT-CUED odor port is reached.
  4. Going to the WRONG odor port does NOT reveal the odor.
  5. Reaching correct reward port -> reward=+1, done=True.
  6. Reaching wrong reward port -> reward=0, done=True (still terminates).
  7. Random policy occasionally solves the task (so a learner has signal).
"""

import numpy as np
from two_cue_gridworld import (
    TwoCueGridWorld, OBS_LIGHT, OBS_ODOR, ODOR_HIDDEN
)


def test_observation_shape():
    env = TwoCueGridWorld(seed=0)
    obs = env.reset()
    assert obs.shape == (4,) and obs.dtype == np.float32
    print("[PASS] obs shape and dtype:", obs.shape, obs.dtype)


def test_cues_change_across_episodes():
    """Across many resets, both cues should take both values."""
    env = TwoCueGridWorld(seed=0)
    lights, odors = set(), set()
    for _ in range(50):
        env.reset()
        lights.add(env.light_cue)
        odors.add(env.odor_cue)
    assert lights == {0, 1}, f"light cue stuck at {lights}"
    assert odors  == {0, 1}, f"odor cue stuck at {odors}"
    print("[PASS] both cues take both values across episodes")


def test_odor_hidden_at_start():
    env = TwoCueGridWorld(seed=0)
    obs = env.reset()
    assert obs[OBS_ODOR] == ODOR_HIDDEN, f"odor leaked at start: {obs[OBS_ODOR]}"
    assert obs[OBS_LIGHT] in (0.0, 1.0)
    print("[PASS] odor hidden at start, light visible")


def _walk_to(env, target):
    """Helper: walk the agent to a target (row, col) using simple greedy moves.
    Returns the final observation."""
    obs = env._obs()
    while not np.array_equal(env.pos, target):
        drow = target[0] - env.pos[0]
        dcol = target[1] - env.pos[1]
        # Pick the action that reduces the larger coord difference.
        if abs(drow) >= abs(dcol) and drow != 0:
            action = 0 if drow < 0 else 2  # up or down
        else:
            action = 1 if dcol > 0 else 3  # right or left
        obs, r, done, trunc, _ = env.step(action)
        if done or trunc:
            return obs, r, done, trunc
    return obs, 0.0, False, False


def test_odor_revealed_at_correct_port():
    # Force a known cue combination by re-seeding until we get what we want.
    for trial_light in (0, 1):
        env = TwoCueGridWorld(seed=0, sniff_bonus=0.2)
        # Spin until light_cue matches what we want to test.
        for _ in range(20):
            env.reset()
            if env.light_cue == trial_light:
                break
        else:
            raise RuntimeError("couldn't get desired light cue")

        cued_port = env._odor_ports[trial_light]
        # Walk to port, but capture the reward on the *arrival* step.
        obs = env._obs()
        sniff_reward = None
        while not np.array_equal(env.pos, cued_port):
            drow = cued_port[0] - env.pos[0]
            dcol = cued_port[1] - env.pos[1]
            if abs(drow) >= abs(dcol) and drow != 0:
                action = 0 if drow < 0 else 2
            else:
                action = 1 if dcol > 0 else 3
            obs, r, done, trunc, _ = env.step(action)
            if env.odor_revealed and sniff_reward is None:
                sniff_reward = r  # capture the step that revealed odor
        assert obs[OBS_ODOR] != ODOR_HIDDEN, \
            f"odor not revealed at correct port (light={trial_light})"
        assert obs[OBS_ODOR] == float(env.odor_cue)
        assert sniff_reward == 0.2, f"expected sniff bonus 0.2, got {sniff_reward}"

        # One-shot check: stepping again at the port should NOT give bonus.
        # Move away and back.
        env.step(2)  # down
        obs, r, _, _, _ = env.step(0)  # back up to port
        assert r == 0.0, f"sniff bonus fired twice (got r={r})"
    print("[PASS] odor revealed + sniff bonus fires once per episode")


def test_wrong_odor_port_no_reveal():
    env = TwoCueGridWorld(seed=0)
    for _ in range(20):
        env.reset()
        if env.light_cue == 0:
            break
    # Light says go to port 0, send agent to port 1 instead.
    wrong_port = env._odor_ports[1]
    obs, _, done, _ = _walk_to(env, wrong_port)
    assert not done
    assert obs[OBS_ODOR] == ODOR_HIDDEN, "odor leaked at wrong port"
    print("[PASS] wrong odor port does not reveal odor")


def test_correct_reward_port():
    env = TwoCueGridWorld(seed=0)
    # Find a (light, odor) combo to test.
    for _ in range(20):
        env.reset()
        break
    target = env._reward_ports[env.odor_cue]
    obs, r, done, _ = _walk_to(env, target)
    assert done and r == 1.0, f"expected (done=True, r=1.0), got ({done}, {r})"
    print(f"[PASS] correct reward port -> r=+1, done=True (light={env.light_cue}, odor={env.odor_cue})")


def test_wrong_reward_port_terminates():
    env = TwoCueGridWorld(seed=0)
    env.reset()
    wrong_port = env._reward_ports[1 - env.odor_cue]
    obs, r, done, _ = _walk_to(env, wrong_port)
    assert done and r == 0.0, f"expected (done=True, r=0.0), got ({done}, {r})"
    print(f"[PASS] wrong reward port -> r=0, done=True (terminates)")


def test_random_policy_baseline(n_episodes=500):
    """Random policy should solve some episodes -- we need reward signal."""
    env = TwoCueGridWorld(seed=0)
    rng = np.random.default_rng(123)
    successes = 0
    revealed_count = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = trunc = False
        revealed_this_ep = False
        while not (done or trunc):
            a = int(rng.integers(0, 4))
            obs, r, done, trunc, _ = env.step(a)
            if obs[OBS_ODOR] != ODOR_HIDDEN:
                revealed_this_ep = True
        if r == 1.0:
            successes += 1
        if revealed_this_ep:
            revealed_count += 1
    succ_rate    = successes / n_episodes
    reveal_rate  = revealed_count / n_episodes
    print(f"[INFO] random policy: success {succ_rate:.1%}, "
          f"odor revealed {reveal_rate:.1%}")
    # Random success should be above zero but well below 50% (which would
    # mean the task is trivial).
    assert 0.02 < succ_rate < 0.5, \
        f"random success rate {succ_rate:.1%} is suspicious"
    print("[PASS] random baseline gives learnable signal")


if __name__ == "__main__":
    test_observation_shape()
    test_cues_change_across_episodes()
    test_odor_hidden_at_start()
    test_odor_revealed_at_correct_port()
    test_wrong_odor_port_no_reveal()
    test_correct_reward_port()
    test_wrong_reward_port_terminates()
    test_random_policy_baseline()
    print("\nAll two-cue env tests passed.")
