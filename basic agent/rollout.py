"""
Stage 4 (sanity): roll out the trained agent greedily and watch it solve the env.
We use argmax instead of sampling here just to see the most-likely behavior.
"""

import torch
import numpy as np
from gridworld import GridWorld
from train import train


def rollout_greedy(agent, env, max_steps=30, render=True):
    obs = env.reset()
    if render:
        print("--- start ---")
        env.render()
    for t in range(max_steps):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            logits, value = agent.net(obs_t)
            action = int(logits.argmax(dim=-1).item())
        action_names = {0: "up", 1: "right", 2: "down", 3: "left"}
        obs, r, done, trunc, _ = env.step(action)
        if render:
            print(f"\nstep {t+1}: action={action_names[action]}, V(s)={value.item():+.3f}, r={r}")
            env.render()
        if done:
            print(f"\n[reached goal in {t+1} steps]")
            return t + 1
        if trunc:
            print("\n[truncated]")
            return None
    return None


if __name__ == "__main__":
    # Re-train with the same seed so we get the same agent we just saw learn.
    print("Re-training (deterministic seed)...\n")
    agent = train()
    print("\n\n=== Greedy rollout of trained agent ===\n")
    env = GridWorld()
    rollout_greedy(agent, env)
