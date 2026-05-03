"""
Stage 3: training loop.

Key structural choice: we collect rollouts of fixed length n_steps, NOT full
episodes. Episodes can end mid-rollout, and that's fine -- the `dones` mask
handles return computation across episode boundaries.

Why fixed-length rollouts? It decouples update frequency from episode length.
With episode-based updates, an agent that wanders for 50 steps would update
much less often than one that finishes in 10. Fixed-length keeps gradient
updates regular.
"""

import numpy as np
import torch
from collections import deque
from gridworld import GridWorld
from agent import A2CAgent


def train(
    total_steps=30_000,
    n_steps=20,
    seed=0,
    log_every=2_000,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = GridWorld()
    agent = A2CAgent(obs_dim=env.obs_dim, n_actions=env.n_actions, n_steps=n_steps)

    obs = env.reset()
    ep_return = 0.0
    ep_length = 0
    # Rolling buffers for diagnostics
    recent_returns = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_successes = deque(maxlen=100)

    step_count = 0
    while step_count < total_steps:
        # ---- Collect a rollout of n_steps ----
        for _ in range(n_steps):
            action = agent.act(obs)
            next_obs, reward, done, trunc, _ = env.step(action)
            agent.store_step(reward, done or trunc)

            ep_return += reward
            ep_length += 1
            step_count += 1

            if done or trunc:
                recent_returns.append(ep_return)
                recent_lengths.append(ep_length)
                recent_successes.append(1.0 if done else 0.0)
                obs = env.reset()
                ep_return = 0.0
                ep_length = 0
            else:
                obs = next_obs

        # ---- Update ----
        # `last_done` here is whether the most recent stored step ended an
        # episode. If yes, bootstrap value = 0; otherwise use V(obs).
        last_done = bool(agent.dones[-1])
        diag = agent.update(obs, last_done)

        # ---- Logging ----
        if step_count % log_every < n_steps and len(recent_returns) > 0:
            print(
                f"step {step_count:6d} | "
                f"success {np.mean(recent_successes):.2f} | "
                f"ep_len {np.mean(recent_lengths):5.1f} | "
                f"return {np.mean(recent_returns):.3f} | "
                f"V(s) {diag['value_mean']:+.3f} | "
                f"adv {diag['advantage_mean']:+.3f}±{diag['advantage_std']:.3f} | "
                f"H {diag['entropy']:.3f} | "
                f"pi_loss {diag['policy_loss']:+.3f} | "
                f"v_loss {diag['value_loss']:.3f} | "
                f"|g| {diag['grad_norm']:.3f}"
            )

    return agent


if __name__ == "__main__":
    agent = train()
