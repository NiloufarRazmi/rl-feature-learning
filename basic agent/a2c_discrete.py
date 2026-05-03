"""
A2C (Advantage Actor-Critic) with Discrete Action Space
========================================================
A clean PyTorch implementation structured for easy extension.

Key differences from DQN:
  - Two output heads: policy (actor) + value (critic)
  - On-policy: rollouts are collected, used once, then discarded
  - No replay buffer, no target network
  - Loss = policy_loss + value_loss + entropy_bonus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from collections import namedtuple

# ──────────────────────────────────────────────
# 1. Network Architecture
# ──────────────────────────────────────────────

class ActorCriticNet(nn.Module):
    """
    Shared-trunk network with two heads:
      - Actor head  → logits over discrete actions (policy π)
      - Critic head → scalar state value V(s)

    Sharing the trunk lets both heads learn a common representation,
    which tends to work well when state features are useful for both
    policy and value estimation (almost always true).
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: outputs unnormalized log-probabilities (logits)
        self.actor_head = nn.Linear(hidden_dim, n_actions)

        # Critic head: outputs a single scalar V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init (common best practice for policy gradient methods)."""
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)  # small init for policy
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x: torch.Tensor):
        """Returns (action_logits, state_value)."""
        features = self.trunk(x)
        logits = self.actor_head(features)
        value  = self.critic_head(features).squeeze(-1)  # shape: (batch,)
        return logits, value

    def get_action(self, obs: torch.Tensor):
        """
        Sample an action from π(·|s), returning:
          action      – sampled action index
          log_prob    – log π(a|s), needed for the policy gradient
          value       – V(s) from the critic
          entropy     – H[π(·|s)], used as a bonus to encourage exploration
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, value, entropy


# ──────────────────────────────────────────────
# 2. Rollout Buffer
# ──────────────────────────────────────────────

Transition = namedtuple("Transition",
    ["obs", "action", "log_prob", "value", "reward", "done"])

class RolloutBuffer:
    """
    Stores one rollout (n_steps of experience) before computing
    returns and updating the network. On-policy: cleared after each update.
    """
    def __init__(self):
        self.transitions: list[Transition] = []

    def push(self, *args):
        self.transitions.append(Transition(*args))

    def clear(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)


# ──────────────────────────────────────────────
# 3. A2C Agent
# ──────────────────────────────────────────────

class A2CAgent:
    """
    Synchronous Advantage Actor-Critic.

    The update rule (per rollout):
      advantage   A(s,a) = R_t - V(s)        (where R_t is the n-step return)
      actor loss  = -E[log π(a|s) * A(s,a)]  (policy gradient)
      critic loss =  E[(R_t - V(s))²]        (value regression)
      total loss  = actor_loss
                  + value_coef * critic_loss
                  - entropy_coef * H[π]      (entropy bonus for exploration)
    """

    def __init__(
        self,
        obs_dim:       int,
        n_actions:     int,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        n_steps:       int   = 5,       # rollout length before each update
        value_coef:    float = 0.5,     # weight on critic loss
        entropy_coef:  float = 0.01,    # weight on entropy bonus
        max_grad_norm: float = 0.5,     # gradient clipping
        hidden_dim:    int   = 128,
        device:        str   = "cpu",
    ):
        self.gamma        = gamma
        self.n_steps      = n_steps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm= max_grad_norm
        self.device       = torch.device(device)

        self.net    = ActorCriticNet(obs_dim, n_actions, hidden_dim).to(self.device)
        self.opt    = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        # Logging
        self.total_steps   = 0
        self.episode_count = 0

    # ── Interaction ──────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        """Called at every env step. Returns numpy action."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, value, entropy = self.net.get_action(obs_t)
        return (
            action.item(),
            log_prob.item(),
            value.item(),
            entropy.item(),
        )

    def store(self, obs, action, log_prob, value, reward, done):
        self.buffer.push(obs, action, log_prob, value, reward, done)

    # ── Return Computation ───────────────────────

    def _compute_returns(self, next_obs: np.ndarray, next_done: bool) -> torch.Tensor:
        """
        Bootstrapped n-step returns (discounted cumulative rewards).
        If the episode ended (done=True), bootstrap value is 0.
        Otherwise we bootstrap with V(s_{t+n}).
        """
        if next_done:
            next_value = 0.0
        else:
            with torch.no_grad():
                obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
                _, next_value = self.net(obs_t)
                next_value = next_value.item()

        returns = []
        R = next_value
        for t in reversed(self.buffer.transitions):
            R = t.reward + self.gamma * R * (1 - float(t.done))
            returns.insert(0, R)

        return torch.FloatTensor(returns).to(self.device)

    # ── Update ───────────────────────────────────

    def update(self, next_obs: np.ndarray, next_done: bool) -> dict:
        """
        Compute A2C loss and backprop. Called after every n_steps transitions.
        Returns a dict of loss components for logging.
        """
        returns = self._compute_returns(next_obs, next_done)

        # Unpack buffer into tensors
        obs_t      = torch.FloatTensor(
                         np.array([t.obs      for t in self.buffer.transitions])
                     ).to(self.device)
        actions_t  = torch.LongTensor(
                         [t.action   for t in self.buffer.transitions]
                     ).to(self.device)
        old_values = torch.FloatTensor(
                         [t.value    for t in self.buffer.transitions]
                     ).to(self.device)

        # Forward pass with grad
        logits, values = self.net(obs_t)
        dist     = Categorical(logits=logits)
        log_probs= dist.log_prob(actions_t)
        entropy  = dist.entropy().mean()

        # Advantage (detach returns so critic gradient doesn't flow into actor)
        advantages = (returns - values.detach())

        # Normalize advantages (stabilizes training)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Losses
        actor_loss  = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        total_loss  = (actor_loss
                       + self.value_coef  * critic_loss
                       - self.entropy_coef * entropy)

        # Backprop
        self.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.opt.step()

        self.buffer.clear()

        return {
            "loss/total":  total_loss.item(),
            "loss/actor":  actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/entropy":entropy.item(),
        }


# ──────────────────────────────────────────────
# 4. Training Loop
# ──────────────────────────────────────────────

def train(
    env_id:        str   = "OdorNavEnv",
    total_steps:   int   = 200_000,
    n_steps:       int   = 5,
    lr:            float = 3e-4,
    gamma:         float = 0.99,
    entropy_coef:  float = 0.01,
    value_coef:    float = 0.5,
    hidden_dim:    int   = 128,
    seed:          int   = 42,
    log_interval:  int   = 1000,
):
    env = gym.make(env_id)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

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
    ep_reward   = 0.0
    ep_rewards  = []
    step        = 0

    print(f"Training A2C on {env_id} | obs_dim={obs_dim} | n_actions={n_actions}")
    print(f"{'Step':>8}  {'Ep':>5}  {'MeanRew(10)':>12}  {'Actor':>9}  {'Critic':>9}  {'Entropy':>9}")
    print("-" * 65)

    while step < total_steps:
        # ── Collect n_steps rollout ──
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

        # ── Update ──
        metrics = agent.update(next_obs=obs, next_done=done)

        # ── Logging ──
        if step % log_interval < n_steps and ep_rewards:
            mean_rew = np.mean(ep_rewards[-10:])
            print(f"{step:>8}  {agent.episode_count:>5}  {mean_rew:>12.2f}  "
                  f"{metrics['loss/actor']:>9.4f}  {metrics['loss/critic']:>9.4f}  "
                  f"{metrics['loss/entropy']:>9.4f}")

    env.close()
    print("\nTraining complete.")
    return agent


# ──────────────────────────────────────────────
# 5. Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    agent = train(
        env_id="OdorNavEnv",
        total_steps=200_000,
        n_steps=5,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        hidden_dim=128,
    )
