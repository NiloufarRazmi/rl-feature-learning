"""
Odor-Cued Navigation Environment
=================================
Python/Gymnasium port of main_task.m

The task:
  - Agent navigates an 11x11 grid (coordinates -1 to 1, step 0.2)
  - Two odor ports and two reward ports are placed in the arena
  - Each trial: agent starts at (0,0), navigates to an odor port to receive
    a cue (odor 1 or 2), then must go to the correct reward port
  - Correct port depends on which odor was received (context mapping)
  - Observation: gaussian bump (place cells) over 121 locations + 4 cue dims
  - Actions: 0=up, 1=down, 2=left, 3=right
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────
# Helpers 
# ──────────────────────────────────────────────

def gaussian_bump(x: float, y: float,
                  grid: np.ndarray = None,
                  sigma: float = 0.2) -> np.ndarray:
    """
    121-dim place cell population code for position (x, y).
    """
    if grid is None:
        locs = np.linspace(-1, 1, 11)
        xs, ys = np.meshgrid(locs, locs)
        grid = np.stack([xs.ravel(), ys.ravel()], axis=1)  # (121, 2)
    dist_sq = (grid[:, 0] - x)**2 + (grid[:, 1] - y)**2
    bump = np.exp(-dist_sq / (2 * sigma**2))
    return bump  # shape (121,)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


# Pre-compute grid once at module level
_LOCS = np.linspace(-1, 1, 11)
_XS, _YS = np.meshgrid(_LOCS, _LOCS)
_GRID = np.stack([_XS.ravel(), _YS.ravel()], axis=1)  # (121, 2)

# Odor and light lookup tables (index 0=odor1, 1=odor2, 2=none)
_ODOR_LIST  = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
_LIGHT_LIST = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class OdorNavEnv(gym.Env):
    """
    Odor-cued allocentric navigation task.

    Parameters
    ----------
    max_reward      : reward magnitude on correct choice (default 1.0)
    max_steps       : max steps per episode before truncation (default 500)
    wall_condition  : 0='vertical', 1='horizontal', 2='no_wall' (default 2)
    random_odor     : randomize which port gives odor (default True)
    random_reward   : randomize reward port locations (default False)
    allocentric     : use allocentric task rules (default True)
    context         : which odor→reward mapping to use, 0 or 1 (default 0)
    seed            : RNG seed
    """

    metadata = {"render_modes": []}

    # Action definitions: (dx, dy)
    _ACTIONS = {
        0: ( 0.0,  0.2),   # up
        1: ( 0.0, -0.2),   # down
        2: (-0.2,  0.0),   # left
        3: ( 0.2,  0.0),   # right
    }

    def __init__(
        self,
        max_reward:     float = 50.0,
        max_steps:      int   = 500,
        wall_condition: int   = 2,
        random_odor:    bool  = True,
        random_reward:  bool  = False,
        allocentric:    bool  = True,
        context:        int   = 0,
        seed:           int   = 0,
    ):
        super().__init__()

        self.max_reward     = max_reward
        self.max_steps      = max_steps
        self.wall_condition = wall_condition
        self.random_odor    = random_odor
        self.random_reward  = random_reward
        self.allocentric    = allocentric
        self.d              = 0.2          # proximity threshold 

        # Odor→reward context mapping (from your odor_vec = [1 2;2 1;1 2;2 1])
        # context 0: odor1→port1, odor2→port2
        # context 1: odor1→port2, odor2→port1
        _odor_vec = [(1, 2), (2, 1)]
        self.context_1, self.context_2 = _odor_vec[context % 2]

        # Observation: 121 place cells + 2 odor dims + 2 light dims = 125
        self.obs_dim = 121 + 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.rng = np.random.default_rng(seed)
        self._setup_ports()

        # Episode state
        self.x = 0.0
        self.y = 0.0
        self.odor_cue        = 2      # 0=odor1, 1=odor2, 2=none 
        self.active_odor_port= 0      # 0 or 1 (which port is active this trial)
        self.step_count      = 0

    # ── Port Setup ──────────────────────────────

    def _setup_ports(self):
        """
        Port positions from your port_setup.m logic.
        wall_condition 2 = no_wall (default).
        """
        wc = self.wall_condition

        if wc == 0:   # vertical wall
            self.reward_port_1 = np.array([-0.7,  0.7])
            self.reward_port_2 = np.array([ 0.7, -0.7])
            self.odor_port_1   = np.array([-0.7, -0.7])
            self.odor_port_2   = np.array([ 0.7,  0.7])
            self.wall          = "vertical"
        elif wc == 1: # horizontal wall
            self.reward_port_1 = np.array([-0.7,  0.7])
            self.reward_port_2 = np.array([ 0.7, -0.7])
            self.odor_port_1   = np.array([ 0.7,  0.7])
            self.odor_port_2   = np.array([-0.7, -0.7])
            self.wall          = "horizontal"
        else:         # no wall
            self.reward_port_1 = np.array([-0.7,  0.7])
            self.reward_port_2 = np.array([ 0.7, -0.7])
            self.odor_port_1   = np.array([-0.7, -0.7])
            self.odor_port_2   = np.array([ 0.7,  0.7])
            self.wall          = "no_wall"

        if self.random_reward:
            corners = [[-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7], [0.7, 0.7]]
            idxs = self.rng.choice(4, size=2, replace=False)
            self.reward_port_1 = np.array(corners[idxs[0]], dtype=float)
            self.reward_port_2 = np.array(corners[idxs[1]], dtype=float)

    def _init_trial(self):
        """Reset position to (0,0) and pick an active odor port."""
        self.x = 0.0
        self.y = 0.0
        self.odor_cue = 2                              # no odor yet
        self.active_odor_port = int(self.rng.integers(0, 2))  # 0 or 1

    # ── Observation ──────────────────────────────

    def _get_obs(self) -> np.ndarray:
        place_cells = gaussian_bump(self.x, self.y, _GRID).astype(np.float32)
        odor_vec    = _ODOR_LIST[self.odor_cue]
        light_vec   = _LIGHT_LIST[self.active_odor_port]
        return np.concatenate([place_cells, odor_vec, light_vec])

    # ── Movement ─────────────────────────────────

    def _move(self, action: int):
        """
        Apply action with wall constraints.
        Equivalent to allo_move_walls.m — clamps position to [-1, 1].
        Wall logic can be extended here if needed.
        """
        dx, dy = self._ACTIONS[action]
        new_x = np.clip(self.x + dx, -1.0, 1.0)
        new_y = np.clip(self.y + dy, -1.0, 1.0)

        # Vertical wall: block crossing x=0 between y in [-0.2, 0.2]
        if self.wall == "vertical":
            if self.x * new_x < 0 and abs(new_y) <= 0.2:
                new_x = self.x  # blocked

        # Horizontal wall: block crossing y=0 between x in [-0.2, 0.2]
        elif self.wall == "horizontal":
            if self.y * new_y < 0 and abs(new_x) <= 0.2:
                new_y = self.y  # blocked

        self.x, self.y = new_x, new_y

    def _near(self, port: np.ndarray) -> bool:
        return abs(port[0] - self.x) < self.d and abs(port[1] - self.y) < self.d

    # ── Gym Interface ────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_trial()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
        self._move(action)
        self.step_count += 1

        reward     = 0.0
        terminated = False

        # ── Odor port check (only if no odor yet) ──
        if self.odor_cue == 2:
            if self.active_odor_port == 0 and self._near(self.odor_port_1):
                self.odor_cue = int(self.rng.integers(0, 2))   # 0 or 1
            elif self.active_odor_port == 1 and self._near(self.odor_port_2):
                self.odor_cue = int(self.rng.integers(0, 2))

        # ── Reward port check (only after receiving odor) ──
        elif self.odor_cue != 2:
            # context_1 and context_2 
            ctx1 = self.context_1 - 1
            ctx2 = self.context_2 - 1

            if self._near(self.reward_port_1):
                if self.odor_cue == ctx1:
                    reward = self.max_reward
                terminated = True

            elif self._near(self.reward_port_2):
                if self.odor_cue == ctx2:
                    reward = self.max_reward
                terminated = True

        truncated = self.step_count >= self.max_steps

        if terminated or truncated:
            self._init_trial()   # ready for next episode

        obs = self._get_obs()
        info = {
            "x": self.x, "y": self.y,
            "odor_cue": self.odor_cue,
            "active_odor_port": self.active_odor_port,
        }
        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    env = OdorNavEnv(max_reward=50.0, seed=42)
    obs, _ = env.reset()
    print(f"obs shape : {obs.shape}")       # should be (125,)
    print(f"action space: {env.action_space}")

    total_reward = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Random agent total reward over 200 steps: {total_reward}")
