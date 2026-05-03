"""
Training A2C on the two-cue gridworld.

The diagnostic that matters most here is success rate BROKEN DOWN BY CUE
COMBINATION. The aggregate success rate can be misleading: a policy that
ignores cues and always goes to R1 gets ~50% success but has learned nothing.
A real learner gets close to 100% on each of the 4 (light, odor) combos.

We also track "odor used" -- the fraction of episodes where the odor was
actually revealed before termination. A solver that skips the odor port
won't generalize.
"""

import numpy as np
import torch
from collections import deque, defaultdict
from two_cue_gridworld import TwoCueGridWorld, OBS_ODOR, ODOR_HIDDEN
from agent import A2CAgent


def train(
    total_steps=1200_000,
    n_steps=20,
    seed=0,
    log_every=4_000,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = TwoCueGridWorld(seed=seed)
    agent = A2CAgent(obs_dim=env.obs_dim, n_actions=env.n_actions, n_steps=n_steps)

    obs = env.reset()
    ep_return = 0.0
    ep_length = 0
    ep_used_odor = False
    # Episode stats keyed by (light, odor)
    ep_light, ep_odor = env.light_cue, env.odor_cue

    # Per-(light, odor) success rolling buffers
    cue_successes = defaultdict(lambda: deque(maxlen=50))
    recent_lengths   = deque(maxlen=200)
    recent_successes = deque(maxlen=200)
    recent_used_odor = deque(maxlen=200)

    step_count = 0
    while step_count < total_steps:
        # ---- Collect a rollout ----
        for _ in range(n_steps):
            action = agent.act(obs)
            next_obs, reward, done, trunc, _ = env.step(action)
            agent.store_step(reward, done or trunc)

            ep_return += reward
            ep_length += 1
            step_count += 1
            if next_obs[OBS_ODOR] != ODOR_HIDDEN:
                ep_used_odor = True

            if done or trunc:
                # Bookkeeping for this episode.
                got_reward = (reward ==0.1)
                cue_successes[(ep_light, ep_odor)].append(1.0 if got_reward else 0.0)
                recent_successes.append(1.0 if got_reward else 0.0)
                recent_lengths.append(ep_length)
                recent_used_odor.append(1.0 if ep_used_odor else 0.0)
                # Reset.
                obs = env.reset()
                ep_return, ep_length, ep_used_odor = 0.0, 0, False
                ep_light, ep_odor = env.light_cue, env.odor_cue
            else:
                obs = next_obs

        # ---- Update ----
        last_done = bool(agent.dones[-1])
        diag = agent.update(obs, last_done)

        # ---- Logging ----
        if step_count % log_every < n_steps and len(recent_successes) > 0:
            # Per-cue success
            cue_str_parts = []
            for (l, o) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                buf = cue_successes[(l, o)]
                if len(buf) > 0:
                    cue_str_parts.append(f"L{l}O{o}={np.mean(buf):.2f}")
                else:
                    cue_str_parts.append(f"L{l}O{o}=  ? ")
            cue_str = " ".join(cue_str_parts)

            print(
                f"step {step_count:6d} | "
                f"succ {np.mean(recent_successes):.2f} | "
                f"odor_used {np.mean(recent_used_odor):.2f} | "
                f"ep_len {np.mean(recent_lengths):5.1f} | "
                f"H {diag['entropy']:.2f} | "
                f"V {diag['value_mean']:+.2f} | "
                f"adv {diag['advantage_mean']:+.3f} | "
                f"|g| {diag['grad_norm']:.2f} || {cue_str}"
            )
    torch.save(agent.net.state_dict(), "trained_agent.pt")
    return agent, env


if __name__ == "__main__":
    train()


