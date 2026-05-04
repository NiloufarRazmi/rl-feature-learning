"""
Stage 2: the actor-critic network.

One shared trunk -> two heads:
  - actor:  outputs logits over n_actions. We sample from Categorical(logits=...).
  - critic: outputs a scalar V(s).

Why one network with two heads (instead of two separate networks)?
  - Features useful for choosing actions are usually useful for predicting values.
  - Half the parameters, faster to train.
  - It's the standard A2C choice. Separating them is an option if you find the
    two losses are fighting each other; for now, share.

Important detail: we return *logits*, not softmax probabilities.
torch.distributions.Categorical(logits=...) handles the softmax internally
in a numerically stable way. If you pass softmax probs and one is ~0, log(0)
will give you nan losses.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=400):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head  = nn.Linear(hidden, n_actions)  # logits
        self.critic_head = nn.Linear(hidden, 1)          # scalar value

        # Small init for the policy head: keeps the initial policy near-uniform,
        # which means lots of exploration at the start. Big init -> the policy
        # commits to whichever action got the largest random weight, and
        # exploration suffers.
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        # Larger init for the value head is fine; it's regression, not a
        # distribution.
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs):
        """obs: (batch, obs_dim) tensor. Returns (logits, value)."""
        h = self.trunk(obs)
        logits = self.actor_head(h)         # (batch, n_actions)
        value  = self.critic_head(h).squeeze(-1)  # (batch,)
        return logits, value

    def act(self, obs):
        """Sample an action and return everything we need for training.

        Returns:
            action:   int, the sampled action
            log_prob: tensor, log pi(a|s)  -- needed for policy loss
            value:    tensor, V(s)         -- needed for advantage
            entropy:  tensor, H(pi(.|s))   -- needed for entropy bonus
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, value, entropy
