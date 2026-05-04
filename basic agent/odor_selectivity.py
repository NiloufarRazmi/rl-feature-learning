"""
Cosine-similarity heatmap analysis.

For each grid location (row, col), compute the cosine similarity between
the network's hidden representation under two different conditions:
  h(loc, light, odor=0)  vs  h(loc, light, odor=1)

Both odor states are taken as REVEALED (i.e. the network "knows" what the
odor is). We do this separately for each light condition, giving two
heatmaps.

Interpretation:
  - cos = 1.0  -> the two odor states map to identical hidden activations.
                  At this location, the network ignores the odor.
  - cos < 1.0  -> the network's representation differs between the two
                  odor states. Lower = more different.

Where do we EXPECT to see selectivity (low cosine)?
  - Near the reward ports, because that's where the action choice depends
    on which odor was sniffed.
  - And probably along the path between odor port and reward port, since
    the network has to "carry" the odor info to act on it.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from network import ActorCritic
from two_cue_gridworld import (
    OBS_ROW, OBS_COL,
    OBS_LIGHT_0, OBS_LIGHT_1,
    OBS_ODOR_0, OBS_ODOR_1,
    OBS_DIM,
)


def load_network(path="trained_agent.pt", obs_dim=OBS_DIM, n_actions=4, hidden=64):
    """Load a trained ActorCritic. `hidden` must match what was trained."""
    net = ActorCritic(obs_dim, n_actions, hidden=hidden)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


def get_hidden(net, obs_batch):
    """Return the second-layer trunk activations for each obs."""
    if isinstance(obs_batch, np.ndarray):
        obs_batch = torch.from_numpy(obs_batch).float()
    with torch.no_grad():
        h = net.trunk(obs_batch).numpy()
    return h


def make_obs(row, col, light, odor, env_size=5):
    """Build a one-hot observation. `odor` in {None, 0, 1} where None means hidden.
    With 2-channel odor, hidden = both channels at 0."""
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[OBS_ROW] = row / (env_size - 1)
    o[OBS_COL] = col / (env_size - 1)
    if light == 0:
        o[OBS_LIGHT_0] = 1.0
    else:
        o[OBS_LIGHT_1] = 1.0
    # Odor: leave both channels at 0 for "hidden", else set one.
    if odor == 0:
        o[OBS_ODOR_0] = 1.0
    elif odor == 1:
        o[OBS_ODOR_1] = 1.0
    # else odor is None -> both channels stay 0 -> "hidden"
    return o


def cosine_sim(a, b, eps=1e-9):
    """Cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def compute_odor_selectivity_grid(net, light, env_size=5):
    """For each (row, col), compute cos( h(odor=0), h(odor=1) ) at fixed light.

    Returns a (env_size, env_size) array of cosine similarities.
    """
    # Build all locations as a batch: 25 obs for odor=0, 25 obs for odor=1.
    obs_o0 = np.stack([make_obs(r, c, light, 0.0, env_size)
                       for r in range(env_size) for c in range(env_size)])
    obs_o1 = np.stack([make_obs(r, c, light, 1.0, env_size)
                       for r in range(env_size) for c in range(env_size)])

    H0 = get_hidden(net, obs_o0)   # (25, 64)
    H1 = get_hidden(net, obs_o1)   # (25, 64)

    # Vectorized cosine: row-wise.
    norms0 = np.linalg.norm(H0, axis=1)
    norms1 = np.linalg.norm(H1, axis=1)
    dots   = (H0 * H1).sum(axis=1)
    cos    = dots / (norms0 * norms1 + 1e-9)  # (25,)

    return cos.reshape(env_size, env_size)


def plot_odor_selectivity(net, env_size=5, save="odor_selectivity.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Compute first to get shared color scale.
    cos_grids = []
    for light in [0, 1]:
        cos_grid = compute_odor_selectivity_grid(net, light, env_size)
        cos_grids.append(cos_grid)
    cos_all = np.concatenate([g.ravel() for g in cos_grids])
    vmin, vmax = cos_all.min(), 1.0  # 1.0 is the natural top of cosine sim

    # Use a divergent-ish colormap so deviations from 1.0 are visible.
    # 'magma_r' inverts magma: low values = bright, high values = dark.
    cmap = "magma_r"

    for li, (light, cos_grid) in enumerate(zip([0, 1], cos_grids)):
        ax = axes[li]
        im = ax.imshow(cos_grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")

        # Annotate each cell with the numerical value (helps you SEE the
        # numbers, not just the color).
        for r in range(env_size):
            for c in range(env_size):
                v = cos_grid[r, c]
                # Pick text color based on background brightness.
                tc = "white" if (vmax - v) / (vmax - vmin + 1e-9) > 0.5 else "black"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        color=tc, fontsize=9)

        # Mark task-relevant locations.
        ax.plot(0, 0,            "rs", ms=18, mew=3, fillstyle="none")  # R0
        ax.plot(env_size-1, 0,   "ms", ms=18, mew=3, fillstyle="none")  # R1
        ax.plot(0, env_size//2,            "co", ms=15, mew=3, fillstyle="none")  # O0
        ax.plot(env_size-1, env_size//2,   "yo", ms=15, mew=3, fillstyle="none")  # O1
        ax.plot(env_size//2, env_size//2, "*",  color="white", ms=15,
                mec="black", mew=1.5)  # start

        ax.set_title(f"light = {light}\n"
                     f"cos( h(odor=0) , h(odor=1) ) per location")
        ax.set_xticks(range(env_size))
        ax.set_yticks(range(env_size))
        ax.set_xlabel("col")
        if li == 0:
            ax.set_ylabel("row")

    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02,
                 label="cosine similarity\n(low = odor matters here)")
    fig.suptitle(
        "Where does odor identity change the network's representation?\n"
        "Bright/yellow = odor-selective.   Dark = odor-invariant.",
        fontsize=12)
    plt.savefig(save, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {save}")
    print(f"  cos range: [{cos_all.min():.3f}, {cos_all.max():.3f}]")
    print(f"  most odor-selective location overall: "
          f"min cos = {cos_all.min():.3f}")


if __name__ == "__main__":
    net = load_network()
    plot_odor_selectivity(net)
    print("Done.")
