"""
Stage 1: A minimal 5x5 gridworld.

Layout:
    .  .  .  .  G       <- goal at (0, 4), reward = +1
    .  .  .  .  .
    .  .  .  .  .
    .  .  .  .  .
    S  .  .  .  .       <- start at (4, 0)

Actions: 0=up, 1=right, 2=down, 3=left
Observation: agent's (row, col) as a length-2 float array, normalized to [0, 1].
Reward: +1 on reaching goal, 0 otherwise. Episode ends on goal or after max_steps.
"""

import numpy as np


class GridWorld:
    def __init__(self, size=5, max_steps=50):
        self.size = size
        self.max_steps = max_steps
        self.start = np.array([size - 1, 0])   # bottom-left
        self.goal  = np.array([0, size - 1])   # top-right
        self.n_actions = 4
        self.obs_dim = 2
        self.odor_port_1 =  np.array([0,0])
        self.odor_port_2 =  np.array([size - 1,size - 1])
        self.odor_cue = 3

        self._deltas = {
            0: np.array([-1,  0]),  # up
            1: np.array([ 0,  1]),  # right
            2: np.array([ 1,  0]),  # down
            3: np.array([ 0, -1]),  # left
        }
        self.reset()

    def reset(self):
        self.pos = self.start.copy()
        self.odor_cue = 3
        self.t = 0
        return self._obs()

    def step(self, action):
        new_pos = self.pos + self._deltas[int(action)]
        # clip to grid (walls = no movement)
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.pos = new_pos

        if np.array_equal(new_pos, self.odor_port_1) and self.odor_cue == 3:
            self.odor_cue =2
        elif np.array_equal(new_pos, self.odor_port_2) and self.odor_cue ==3:
            self.odor_cue = 1


        self.t += 1
        end_reward = bool(np.array_equal(self.pos, self.goal) and self.odor_cue !=3)
        end_no_reward = bool(np.array_equal(self.pos, self.goal) and self.odor_cue ==3)
        done = end_reward or end_no_reward
        reward = 1.0 if end_reward else 0.0
        truncated = (self.t >= self.max_steps) and not done
        return self._obs(), reward, done, truncated,{}

    def _obs(self):
        # Normalize to [0, 1] so the network sees nicely-scaled inputs.
        return self.pos.astype(np.float32) / (self.size - 1)

    def render(self):
        grid = np.full((self.size, self.size), ".", dtype=str)
        grid[tuple(self.goal)] = "G"
        grid[tuple(self.pos)] = "A"
        print("\n".join(" ".join(row) for row in grid))
