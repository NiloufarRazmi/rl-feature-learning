"""
Analyze the trained network's internal representations.


1. VALUE LANDSCAPE: V(s) heatmap across (row, col) for each task condition.
   This is "what does the agent think is good" -- the most direct readout
   of what it has learned.

2. PCA OF HIDDEN ACTIVATIONS: project the 64-dim representation down to 2D.
   Color by task variables. Tells us what the geometry encodes.

3. LINEAR DECODING: train tiny logistic regressions from hidden -> task var.
   Tells us what info is LINEARLY accessible from the representation.

4. RDM (Representational Dissimilarity Matrix): pairwise distance matrix
   of activations, blocked by task condition. The standard tool from
   neural data analysis -- you'll be familiar with it.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from network import ActorCritic
from two_cue_gridworld import (
    OBS_ROW, OBS_COL,
    OBS_LIGHT_0, OBS_LIGHT_1,
    OBS_ODOR_0, OBS_ODOR_1,
    OBS_DIM,
)


def load_network(path="trained_agent.pt", obs_dim=OBS_DIM, n_actions=4, hidden=400):
    net = ActorCritic(obs_dim, n_actions, hidden=hidden)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


def hidden_and_value(net, obs_batch):
    """Forward pass that returns hidden activations AND value estimate."""
    if isinstance(obs_batch, np.ndarray):
        obs_batch = torch.from_numpy(obs_batch).float()
    with torch.no_grad():
        h = net.trunk(obs_batch)            # (N, hidden)
        v = net.critic_head(h).squeeze(-1)  # (N,)
        logits = net.actor_head(h)          # (N, 4)
    return h.numpy(), v.numpy(), logits.numpy()


def make_state_grid(env_size=5):
    """Every (row, col, light, odor_state) combo.

    odor_state is a label in {"hidden", "0", "1"}.
    With 2-channel odor: hidden = both channels off.
    """
    obs_list, meta = [], []
    for row in range(env_size):
        for col in range(env_size):
            for light in [0, 1]:
                for odor_label in ["hidden", "0", "1"]:
                    o = np.zeros(OBS_DIM, dtype=np.float32)
                    o[OBS_ROW] = row / (env_size - 1)
                    o[OBS_COL] = col / (env_size - 1)
                    if light == 0:
                        o[OBS_LIGHT_0] = 1.0
                    else:
                        o[OBS_LIGHT_1] = 1.0
                    # Odor: leave both at 0 for "hidden", else set one channel.
                    if odor_label == "0":
                        o[OBS_ODOR_0] = 1.0
                    elif odor_label == "1":
                        o[OBS_ODOR_1] = 1.0
                    # "hidden" -> leave both odor channels at 0
                    obs_list.append(o)
                    meta.append({
                        "row": row, "col": col, "light": light,
                        "odor_state": odor_label,
                        "odor_revealed": (odor_label != "hidden"),
                    })
    return np.stack(obs_list), meta


def plot_value_heatmaps(net, env_size=11, save="value_heatmaps.png"):
    """Six heatmaps: 2 lights x 3 odor states."""
    obs, meta = make_state_grid(env_size)
    _, V, _ = hidden_and_value(net, obs)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    odor_states = ["hidden", "0", "1"]
    odor_labels = ["odor hidden", "odor=0 revealed", "odor=1 revealed"]

    # Find shared color scale for fair comparison across panels.
    vmin, vmax = V.min(), V.max()

    for li, light in enumerate([0, 1]):
        for oi, odor_state in enumerate(odor_states):
            ax = axes[li, oi]
            grid = np.full((env_size, env_size), np.nan)
            for v, m in zip(V, meta):
                if m["light"] == light and m["odor_state"] == odor_state:
                    grid[m["row"], m["col"]] = v
            im = ax.imshow(grid, cmap="viridis", origin="upper",
                           vmin=vmin, vmax=vmax)
            ax.set_title(f"light={light}, {odor_labels[oi]}")

            # Mark task locations. R0 top-left, R1 top-right.
            ax.plot(0, 0, "rs", ms=14, mew=2.5, fillstyle="none")
            ax.plot(env_size-1, 0, "ms", ms=14, mew=2.5, fillstyle="none")
            ax.plot(0, env_size//2, "co", ms=12, mew=2.5, fillstyle="none")
            ax.plot(env_size-1, env_size//2, "yo", ms=12, mew=2.5, fillstyle="none")
            ax.plot(env_size//2, env_size//2, "w*", ms=14, mec="black", mew=1.5)

            ax.set_xticks(range(env_size))
            ax.set_yticks(range(env_size))

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="V(s)")
    fig.suptitle(
        "V(s) across the grid — red squares=R0/R1, circles=O0/O1, star=start",
        fontsize=13)
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {save}  (V range: {vmin:.2f} .. {vmax:.2f})")


def plot_pca(net, save="hidden_pca.png"):
    obs, meta = make_state_grid()
    H, _, _ = hidden_and_value(net, obs)

    pca = PCA(n_components=2)
    H2 = pca.fit_transform(H)
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    light_arr = np.array([m["light"] for m in meta])
    ax = axes[0]
    for L, marker, color in [(0, "o", "C0"), (1, "x", "C3")]:
        idx = light_arr == L
        ax.scatter(H2[idx, 0], H2[idx, 1], c=color, marker=marker, s=24,
                   alpha=0.7, label=f"light={L}")
    ax.legend(); ax.set_title("colored by light cue")

    odor_arr = np.array([m["odor_state"] for m in meta])
    ax = axes[1]
    for label, marker, color, name in [
        ("hidden", "s", "gray", "hidden"),
        ("0", "o", "C0", "odor=0"),
        ("1", "^", "C3", "odor=1"),
    ]:
        idx = odor_arr == label
        ax.scatter(H2[idx, 0], H2[idx, 1], c=color, marker=marker, s=24,
                   alpha=0.7, label=name)
    ax.legend(); ax.set_title("colored by odor state")

    row_arr = np.array([m["row"] for m in meta])
    ax = axes[2]
    sc = ax.scatter(H2[:, 0], H2[:, 1], c=row_arr, cmap="plasma", s=24, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="row (0=top)")
    ax.set_title("colored by grid row")

    for ax in axes:
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
    fig.suptitle("PCA of trunk hidden activations across all states",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {save}  (PC1+PC2 explain "
          f"{(var[0]+var[1])*100:.0f}% of variance)")


def linear_decoding(net):
    obs, meta = make_state_grid()
    H, _, _ = hidden_and_value(net, obs)

    targets = {
        "light":           np.array([m["light"] for m in meta]),
        "odor_revealed":   np.array([m["odor_revealed"] for m in meta]),
        "odor_state":      np.array([m["odor_state"] for m in meta]),
        "row":             np.array([m["row"] for m in meta]),
        "col":             np.array([m["col"] for m in meta]),
    }
    chance = {
        "light": 0.5, "odor_revealed": 2/3, "odor_state": 1/3,
        "row": 0.2, "col": 0.2,
    }
    print("\n  Linear decoding from 64-d hidden activations:")
    print("  variable          accuracy        chance   interpretation")
    print("  -------------------------------------------------------")
    for name, y in targets.items():
        clf = LogisticRegression(max_iter=3000)
        scores = cross_val_score(clf, H, y, cv=5)
        acc, sd = scores.mean(), scores.std()
        ch = chance[name]
        diff = acc - ch
        if diff > 0.30:    verdict = "STRONG"
        elif diff > 0.10:  verdict = "moderate"
        elif diff > 0.03:  verdict = "weak"
        else:              verdict = "near chance"
        print(f"  {name:16s}  {acc:.2f} +/- {sd:.2f}    "
              f"{ch:.2f}     {verdict}")


def plot_rdm(net, save="rdm.png"):
    obs, meta = make_state_grid()
    H, _, _ = hidden_and_value(net, obs)

    conds, cond_means = [], []
    for light in [0, 1]:
        for odor_state, name in [("hidden", "hid"), ("0", "O0"), ("1", "O1")]:
            idx = [i for i, m in enumerate(meta)
                   if m["light"] == light and m["odor_state"] == odor_state]
            cond_means.append(H[idx].mean(axis=0))
            conds.append(f"L{light}/{name}")
    cond_means = np.stack(cond_means)

    norms = np.linalg.norm(cond_means, axis=1, keepdims=True)
    cos_sim = (cond_means @ cond_means.T) / (norms @ norms.T + 1e-9)
    rdm = 1 - cos_sim

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rdm, cmap="magma")
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=45)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds)
    ax.set_title("RDM of hidden representations\n(cosine distance, position-averaged)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {save}")


if __name__ == "__main__":
    net = load_network()
    print("[1/4] Value heatmaps...")
    plot_value_heatmaps(net)
    print("[2/4] PCA of hidden activations...")
    plot_pca(net)
    print("[3/4] Linear decoding...")
    linear_decoding(net)
    print("[4/4] RDM...")
    plot_rdm(net)
    print("\nDone.")
