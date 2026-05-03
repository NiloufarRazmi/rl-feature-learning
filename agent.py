"""
Stage 3: the A2C agent.

Algorithm:
  loop:
    1. Roll out the policy for n_steps in the env, storing
       (obs, action, reward, log_prob, value, done) at each step.
    2. Bootstrap: if the rollout ended without `done`, use V(s_last) as the
       return target for the last step; otherwise 0.
    3. Walk backwards to compute discounted returns:
            R_t = r_t + gamma * R_{t+1}    (with R reset to 0 at episode boundaries)
    4. Advantage:  A_t = R_t - V(s_t)
    5. Loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
       where:
         policy_loss = -mean( log_prob * A.detach() )
         value_loss  =  mean( (R - V(s))^2 )
       The .detach() is critical: gradients through A would couple the two
       losses in a way the algorithm doesn't intend.
    6. backward + step.

Hyperparameters that matter (and what each one does):
  gamma         (0.99): how much we care about future rewards
  n_steps       (20):   rollout length before each update
                        - shorter -> more frequent updates, more bias
                        - longer  -> less bias, more variance, more memory
  value_coef    (0.5):  weight on value loss in the combined objective
  entropy_coef  (0.01): exploration regularization. Too high -> doesn't commit.
                        Too low -> premature convergence to a bad policy.
  lr            (3e-4): standard transformer/PPO default. Surprisingly robust.
  max_grad_norm (0.5):  clip gradients. Prevents single bad updates from
                        nuking the policy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network import ActorCritic


class A2CAgent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        gamma=0.99,
        n_steps=20,
        lr=1e-3,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        hidden=64,
        device="cpu",
    ):
        self.gamma = gamma
        self.n_steps = n_steps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.net = ActorCritic(obs_dim, n_actions, hidden=hidden).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # Rollout storage. We use plain lists and only stack at update time.
        self._reset_rollout()

    def _reset_rollout(self):
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.entropies = []
        self.dones     = []   # 1 if step t ended an episode, else 0

    def act(self, obs_np):
        """Run policy on a single observation. Stores log_prob, value, entropy
        for the upcoming update. Returns the int action."""
        obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
        action, log_prob, value, entropy = self.net.act(obs)
        # Store the per-step quantities (squeeze out the batch dim of 1).
        self.log_probs.append(log_prob.squeeze(0))
        self.values.append(value.squeeze(0))
        self.entropies.append(entropy.squeeze(0))
        return action

    def store_step(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(float(done))

    def update(self, last_obs_np, last_done):
        """Compute returns, advantages, losses and take one optimizer step.

        last_obs_np: the observation AFTER the final stored step. We need it
                     to bootstrap V(s_last) if the rollout was cut short.
        last_done:   whether the final stored step ended an episode. If so,
                     the bootstrap value is 0.
        """
        # ---- 1. Bootstrap value for the step beyond the rollout ----
        with torch.no_grad():
            if last_done:
                bootstrap_value = 0.0
            else:
                last_obs = torch.from_numpy(last_obs_np).float().unsqueeze(0).to(self.device)
                _, v = self.net(last_obs)
                bootstrap_value = v.item()

        # ---- 2. Compute discounted returns, walking backwards ----
        # R_t = r_t + gamma * R_{t+1} * (1 - done_t)
        # The (1 - done) factor zeroes out the bootstrap across episode boundaries:
        # if step t ended the episode, then R_{t+1} should not flow back into R_t.
        returns = []
        R = bootstrap_value
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + self.gamma * R * (1.0 - d)
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # ---- 3. Stack the stored tensors ----
        log_probs = torch.stack(self.log_probs)   # (T,)
        values    = torch.stack(self.values)      # (T,)
        entropies = torch.stack(self.entropies)   # (T,)

        # ---- 4. Advantages ----
        # A_t = R_t - V(s_t). Detached for the policy loss; the value loss
        # uses the un-detached `values` so gradients flow into the critic.
        advantages = returns - values.detach()

        # ---- 5. Losses ----
        policy_loss  = -(log_probs * advantages).mean()
        value_loss   = (returns - values).pow(2).mean()
        entropy_loss = -entropies.mean()  # minimizing this == maximizing entropy

        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # ---- 6. Backprop ----
        self.opt.zero_grad()
        loss.backward()
        # Gradient clipping: a single huge gradient (e.g. from a rare big
        # advantage) can destroy the policy. Clip the global L2 norm.
        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.opt.step()

        diagnostics = {
            "policy_loss":  policy_loss.item(),
            "value_loss":   value_loss.item(),
            "entropy":      entropies.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std":  advantages.std().item() if len(advantages) > 1 else 0.0,
            "value_mean":   values.mean().item(),
            "return_mean":  returns.mean().item(),
            "grad_norm":    grad_norm.item(),
        }
        self._reset_rollout()
        return diagnostics
