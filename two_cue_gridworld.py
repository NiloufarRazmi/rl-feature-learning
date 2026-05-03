"""
Two-cue gridworld task.

Layout (5x5):
    R1  .  .  .  R2     row 0  <- reward ports
     .  .  .  .  .      row 1
    O1  .  S  .  O2     row 2  <- odor ports + start (center)
     .  .  .  .  .      row 3
     .  .  .  .  .      row 4

Trial structure:
  1. Light cue L in {0, 1} drawn uniformly, observable from step 0.
  2. Odor cue O in {0, 1} drawn uniformly, but HIDDEN until agent reaches
     the light-cued odor port.
  3. Light L selects which odor port is "active":
        L = 0 -> O1 (left),  L = 1 -> O2 (right)
  4. Odor O selects which reward port is correct:
        O = 0 -> R1 (left),  O = 1 -> R2 (right)
  5. Reward: +1 at correct reward port (terminal), 0 at wrong reward port
     (also terminal). Wrong odor port = no-op (episode continues).

Observation (length 4):
    [row_norm, col_norm, light_cue, odor_cue]
  - row_norm, col_norm: position normalized to [0, 1]
  - light_cue: 0 or 1, always visible
  - odor_cue: -1 if not yet revealed, else 0 or 1

The "-1 means not revealed" trick is the simplest sentinel encoding. The
network learns to treat -1 as "ignore" by virtue of seeing many examples
where the optimal pre-reveal action is to head toward the odor port (the
odor channel is uninformative in those states).
"""

import numpy as np


# Indices into the observation vector, named for clarity.
# Using named constants beats magic numbers when the obs grows.
OBS_ROW   = 0
OBS_COL   = 1
OBS_LIGHT = 2
OBS_ODOR  = 3
OBS_DIM   = 4

# Sentinel for "odor not yet revealed".
ODOR_HIDDEN = -1.0


class TwoCueGridWorld:
    def __init__(self, size=11, max_steps=200, seed=None, sniff_bonus=0.01):
        assert size >= 5, "layout assumes size >= 5"
        self.size = size
        self.max_steps = max_steps
        self.sniff_bonus = sniff_bonus  # reward for revealing the correct odor; 0.0 disables
        self.rng = np.random.default_rng(seed)

        # Fixed locations on the grid. (row, col) pairs.
        # Using arrays so we can compare with np.array_equal.
        center_col = size // 2
        self.start         = np.array([5,  5])  # middle
        self.odor_port_0   = np.array([0, size - 1])           # left
        self.odor_port_1   = np.array([size - 1,  0])    # right
        self.reward_port_0 = np.array([0,          0])           # top-left
        self.reward_port_1 = np.array([size - 1,          size - 1])    # top-right

        # Mapping cue -> location. Indexable by the cue value (0 or 1),
        # which is much cleaner than `if cue == 0: ... elif cue == 1: ...`.
        self._odor_ports   = [self.odor_port_0,   self.odor_port_1]
        self._reward_ports = [self.reward_port_0, self.reward_port_1]

        self.n_actions = 4
        self.obs_dim   = OBS_DIM

        # action -> (drow, dcol)
        self._deltas = {
            0: np.array([-1,  0]),  # up
            1: np.array([ 0,  1]),  # right
            2: np.array([ 1,  0]),  # down
            3: np.array([ 0, -1]),  # left
        }

        self.reset()

    def reset(self):
        self.pos = self.start.copy()
        self.t = 0
        # Sample new cues for this episode.
        self.light_cue = int(self.rng.integers(0, 2))   # 0 or 1
        self.odor_cue  = int(self.rng.integers(0, 2))   # 0 or 1
        # Odor is hidden until agent reaches the correct odor port.
        self.odor_revealed = False
        return self._obs()

    def step(self, action):
        # Move and clip to grid bounds (walls block motion).
        new_pos = self.pos + self._deltas[int(action)]
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.pos = new_pos
        self.t += 1

        reward = 0.0
        done = False

        # --- Check if agent is at the light-cued odor port: reveal odor. ---
        cued_odor_port = self._odor_ports[self.light_cue]
        if (not self.odor_revealed) and np.array_equal(self.pos, cued_odor_port):
            self.odor_revealed = True
            # Shaping reward: small bonus for sniffing the correct odor.
            # ONE-SHOT (the `not self.odor_revealed` guard above ensures this
            # branch fires at most once per episode). Without that guard, the
            # agent could farm reward by camping at the port.
            reward += self.sniff_bonus

        # --- Check if agent is at any reward port: episode ends. ---
        # We end the episode at EITHER reward port (correct or wrong) so
        # the agent can't just pick randomly until it gets lucky.
        correct_reward_port = self._reward_ports[self.odor_cue]
        for i, port in enumerate(self._reward_ports):
            if np.array_equal(self.pos, port) and self.odor_revealed==True:
                done = True
                if i == self.odor_cue:
                    reward =0.1
                else:
                    reward = 0.0  # could be -1 to discourage early bailing
                break

        truncated = (self.t >= self.max_steps) and not done
        return self._obs(), reward, done, truncated, {}

    def _obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[OBS_ROW]   = self.pos[0] / (self.size - 1)
        obs[OBS_COL]   = self.pos[1] / (self.size - 1)
        obs[OBS_LIGHT] = float(self.light_cue)
        obs[OBS_ODOR]  = float(self.odor_cue) if self.odor_revealed else ODOR_HIDDEN
        return obs

    def render(self):
        """Print a textual view of the grid. Useful for sanity checks."""
        grid = np.full((self.size, self.size), ".", dtype=str)
        # Mark special locations (will be overwritten by 'A' if agent is there).
        grid[tuple(self.reward_port_0)] = "1" if self.odor_cue == 0 else "x"
        grid[tuple(self.reward_port_1)] = "1" if self.odor_cue == 1 else "x"
        grid[tuple(self.odor_port_0)]   = "o"
        grid[tuple(self.odor_port_1)]   = "o"
        grid[tuple(self.pos)]           = "A"
        odor_str = str(self.odor_cue) if self.odor_revealed else "?"
        print(f"light={self.light_cue}, odor={odor_str}, t={self.t}")
        print("\n".join(" ".join(row) for row in grid))
