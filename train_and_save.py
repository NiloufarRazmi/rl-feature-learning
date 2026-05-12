"""
Train an A2C agent on the two-cue task and save it for later analysis.

Uses sniff_bonus=0.2, wrong_port_penalty=-0.5, entropy annealing.
"""

import torch
import numpy as np
from collections import deque, defaultdict
from two_cue_gridworld import TwoCueGridWorld, OBS_ODOR_0, OBS_ODOR_1
from agent import A2CAgent


def entropy_schedule(step, total_steps, start=0.10, end=0.01):
    frac = min(step / total_steps, 1.0)
    return start + frac * (end - start)


def train_and_save(
    total_steps=2000_000,
    n_steps=20,
    seed=0,
    save_path="trained_agent.pt",
    log_every=4_000,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = TwoCueGridWorld(seed=seed, size = 11,max_steps=200, sniff_bonus=5, wrong_port_penalty=0)
    agent = A2CAgent(
        obs_dim=env.obs_dim, n_actions=env.n_actions, n_steps=n_steps,
        entropy_coef=0.10,
    )

    obs = env.reset()
    ep_used_odor = False
    ep_light, ep_odor = env.light_cue, env.odor_cue
    cue_succ = defaultdict(lambda: deque(maxlen=50))
    recent_succ    = deque(maxlen=200)
    recent_used    = deque(maxlen=200)
    recent_lengths = deque(maxlen=200)
    ep_length = 0

    step_count = 0
    while step_count < total_steps:
        agent.entropy_coef = entropy_schedule(step_count, total_steps)

        for _ in range(n_steps):
            action = agent.act(obs)
            next_obs, reward, done, trunc, _ = env.step(action)
            agent.store_step(reward, done or trunc)
            step_count += 1
            ep_length += 1
            if (next_obs[OBS_ODOR_0] + next_obs[OBS_ODOR_1]) > 0.5:  # odor revealed
                ep_used_odor = True

            if done or trunc:
                got = (reward ==50)  # +1 for correct, but with sniff bonus could be ~1.2
                cue_succ[(ep_light, ep_odor)].append(1.0 if got else 0.0)
                recent_succ.append(1.0 if got else 0.0)
                recent_used.append(1.0 if ep_used_odor else 0.0)
                recent_lengths.append(ep_length)
                obs = env.reset()
                ep_used_odor = False
                ep_length = 0
                ep_light, ep_odor = env.light_cue, env.odor_cue
            else:
                obs = next_obs

        last_done = bool(agent.dones[-1])
        diag = agent.update(obs, last_done)

        if step_count % log_every < n_steps and len(recent_succ) > 0:
            cue_str = " ".join(
                f"L{l}O{o}={np.mean(cue_succ[(l, o)]):.2f}"
                for (l, o) in [(0, 0), (0, 1), (1, 0), (1, 1)]
                if len(cue_succ[(l, o)]) > 0
            )
            print(f"step {step_count:7d} | succ {np.mean(recent_succ):.2f} | "
                  f"odor_used {np.mean(recent_used):.2f} | "
                  f"ep_len {np.mean(recent_lengths):4.1f} | "
                  f"H {diag['entropy']:.2f} | {cue_str}")

    torch.save(agent.net.state_dict(), "trained_agent.pt")
    return agent, env


if __name__ == "__main__":
    train_and_save()
