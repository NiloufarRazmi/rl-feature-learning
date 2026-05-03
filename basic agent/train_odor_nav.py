"""
Train A2C on the Odor Navigation Task
======================================
Drop-in replacement for the CartPole demo — just swaps the environment.
"""

from odor_nav_env import OdorNavEnv
from a2c_discrete import A2CAgent
import numpy as np


def train_odor_nav(
    total_steps:   int   =20000,
    n_steps:       int   = 5000,
    lr:            float = 0.005,      
    gamma:         float = 0.9,
    entropy_coef:  float = 0.01,
    value_coef:    float = 0.5,
    hidden_dim:    int   = 128,
    max_reward:    float = 100.0,
    wall_condition:int   = 2,           # 0=vertical, 1=horizontal, 2=no_wall
    context:       int   = 0,
    seed:          int   = 42,
    log_interval:  int   = 2000,
):
    env = OdorNavEnv(
        max_reward=max_reward,
        wall_condition=wall_condition,
        context=context,
        seed=seed,
    )

    obs_dim   = env.obs_dim          # 125 (121 place cells + 4 cue dims)
    n_actions = env.action_space.n   # 4

    agent = A2CAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=lr,
        gamma=gamma,
        n_steps=n_steps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        hidden_dim=hidden_dim,
    )

    obs, _ = env.reset(seed=seed)
    ep_reward  = 0.0
    ep_rewards = []
    step       = 0

    print(f"Training A2C on OdorNavEnv | obs_dim={obs_dim} | n_actions={n_actions}")
    print(f"{'Step':>8}  {'Ep':>5}  {'MeanRew(10)':>12}  {'Actor':>9}  {'Critic':>9}  {'Entropy':>9}")
    print("-" * 65)

    while step < total_steps:
        for _ in range(n_steps):
            action, log_prob, value, entropy = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, log_prob, value, reward, done)
            ep_reward += reward
            step      += 1
            obs        = next_obs

            if done:
                ep_rewards.append(ep_reward)
                agent.episode_count += 1
                ep_reward = 0.0
                obs, _ = env.reset()

        metrics = agent.update(next_obs=obs, next_done=done)

        if step % log_interval < n_steps and ep_rewards:
            mean_rew = np.mean(ep_rewards[-10:])
            print(f"{step:>8}  {agent.episode_count:>5}  {mean_rew:>12.4f}  "
                  f"{metrics['loss/actor']:>9.4f}  {metrics['loss/critic']:>9.4f}  "
                  f"{metrics['loss/entropy']:>9.4f}")

    print("\nDone.")
    return agent


if __name__ == "__main__":
    agent = train_odor_nav(
        total_steps=500_000,
        lr=0.005,
        gamma=0.99,
        hidden_dim=128,
        max_reward=50.0,
        wall_condition=2,
        context=0,
        seed=42,
    )
