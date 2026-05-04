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

Observation vector (length 7), one-hot encoded for the categorical cues:
    [0]: row_normalized
    [1]: col_normalized
    [2]: light == 0 ?           ┐
    [3]: light == 1 ?           ┘ exactly one of these is 1
    [4]: odor not yet revealed? ┐
    [5]: revealed odor == 0?    │ exactly one of these is 1
    [6]: revealed odor == 1?    ┘

Observation vector (length 6):
    [0]: row_normalized
    [1]: col_normalized
    [2]: light == 0 ?     ┐
    [3]: light == 1 ?     ┘ exactly one of these is 1
    [4]: odor == 0 ?      ┐ both 0 means "odor not yet revealed";
    [5]: odor == 1 ?      ┘ otherwise exactly one is 1

Why one-hot? With a scalar like -1/0/1, the network has to LEARN that the
three values are categorically different. With one-hot, the categorical
structure is built into the input geometry: each state is an orthogonal
basis vector. Faster learning, cleaner representations.

Why no explicit "odor hidden" channel? "All-zeros means absent" matches how
sensory features naturally work: a neuron tuned to odor A is silent when
no odor is present, not active in some special "no-odor" state. The network
detects "no odor revealed yet" as the absence of either odor channel.
"""

import numpy as np


# Indices into the observation vector, named for clarity.
OBS_ROW           = 0
OBS_COL           = 1
OBS_LIGHT_0       = 2
OBS_LIGHT_1       = 3
OBS_ODOR_0        = 4
OBS_ODOR_1        = 5
OBS_DIM           = 6


class TwoCueGridWorld:
    def __init__(self, size=5, max_steps=50, seed=None,
                 sniff_bonus=0.1, wrong_port_penalty=0):
        assert size >= 5, "layout assumes size >= 5"
        self.size = size
        self.max_steps = max_steps
        self.sniff_bonus = sniff_bonus
        self.wrong_port_penalty = wrong_port_penalty  # 0.0 disables
        self.rng = np.random.default_rng(seed)

        # Fixed locations on the grid. (row, col) pairs.
        # Using arrays so we can compare with np.array_equal.
        center_col = size // 2
        self.start         = np.array([5,  5])  # middle
        self.odor_port_0   = np.array([size -1,  0])           # left
        self.odor_port_1   = np.array([0,  size - 1])    # right
        self.reward_port_0 = np.array([0, 0])           # top-left
        self.reward_port_1 = np.array([ size - 1, size - 1])    # top-right

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
            if np.array_equal(self.pos, port):
                done = True
                if i == self.odor_cue:
                    reward +=1
                else:
                    reward += self.wrong_port_penalty
                break

        truncated = (self.t >= self.max_steps) and not done
        return self._obs(), reward, done, truncated, {}

    def _obs(self):
        """Build the one-hot observation. See module docstring for layout."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        # Position: continuous, normalized to [0, 1].
        obs[OBS_ROW] = self.pos[0] / (self.size - 1)
        obs[OBS_COL] = self.pos[1] / (self.size - 1)
        # Light cue: one-hot, always set.
        if self.light_cue == 0:
            obs[OBS_LIGHT_0] = 1.0
        else:
            obs[OBS_LIGHT_1] = 1.0
        # Odor cue: 2 channels. Both 0 means "not yet revealed".
        if self.odor_revealed:
            if self.odor_cue == 0:
                obs[OBS_ODOR_0] = 1.0
            else:
                obs[OBS_ODOR_1] = 1.0
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
