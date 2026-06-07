"""Render the optimal pull/save strategy (Objective B) as heatmaps.

Reads the CSVs emitted by gacha.exe and produces, for a given (B, W):
  * decision maps over (pity k, budget b) for g=0 (no guarantee) and g=1
    (next 5-star guaranteed featured), with the threshold curve k*(b) overlaid,
  * the value-function heatmap,
  * the threshold curves k*(b) for both guarantee states.

Usage:  python plot_strategy.py [Btag Wtag]   (default: 180 94)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

N = 90            # hard pity
SOFT = 73         # soft-pity onset (k=73 is the 74th pull)


def load(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def threshold(dec):
    """First pity k (per budget row) at which PULL is optimal; NaN if never."""
    out = np.full(dec.shape[0], np.nan)
    for i in range(dec.shape[0]):
        idx = np.where(dec[i] == 1)[0]
        if idx.size:
            out[i] = idx[0]
    return out


def draw_decision(ax, dec, B, title, big=False):
    lab = 18 if big else 11
    leg = 15 if big else 8
    tick = 15 if big else 10
    budgets = np.arange(1, B + 1)
    cmap = ListedColormap(["#e9ecef", "#2a9d8f"])  # SAVE=gray, PULL=teal
    ax.imshow(dec, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=[0, N - 1, 1, B], interpolation="nearest")
    # threshold curve k*(b)
    ks = threshold(dec)
    ax.plot(ks, budgets, color="#e63946", lw=3.0 if big else 2.0,
            label="threshold k*(b)")
    ax.axvline(SOFT, color="#264653", ls="--", lw=1.6 if big else 1.2,
               label="soft pity (k=73)")
    ax.set_xlabel("pity counter k", fontsize=lab)
    ax.set_ylabel("budget b (pulls left)", fontsize=lab)
    ax.tick_params(labelsize=tick)
    if title:
        ax.set_title(title, fontsize=lab)
    ax.legend(loc="lower left", fontsize=leg, framealpha=0.9)


def draw_value(ax, val, B, title):
    im = ax.imshow(val, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, N - 1, 1, B], interpolation="nearest")
    ax.axvline(SOFT, color="w", ls="--", lw=1.0)
    ax.set_xlabel("pity counter k")
    ax.set_ylabel("budget b (pulls left)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="V (expected net value, pulls)")


def draw_decision_full(dec, B, Bt, Wt, fstate, out):
    """Render a single decision map as its own full-screen figure."""
    fig, ax = plt.subplots(figsize=(16, 9))
    if fstate == 0:
        sub = "f=0  (no guarantee / xiaobaodi)\nteal = PULL, gray = SAVE"
    else:
        sub = "f=1  (next 5-star guaranteed featured / dabaodi)\nteal = PULL, gray = SAVE"
    fig.suptitle("Optimal pull/save decision map  -  {}\n"
                 "budget B={}, featured worth W={} pulls, "
                 "Genshin 5-star pity, classic 50/50 (q=0.5), cost=1 pull/pull"
                 .format(sub, Bt, Wt), fontsize=20)
    draw_decision(ax, dec, B, "", big=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=140)
    print("wrote", out)
    plt.close(fig)


def main():
    Bt = sys.argv[1] if len(sys.argv) > 1 else "180"
    Wt = sys.argv[2] if len(sys.argv) > 2 else "94"
    tag = "B{}_W{}".format(Bt, Wt)
    B = int(Bt)

    dec0 = load("decision_f0_{}.csv".format(tag))
    dec1 = load("decision_f1_{}.csv".format(tag))
    val0 = load("value_f0_{}.csv".format(tag))

    # Two standalone full-screen decision maps.
    draw_decision_full(dec0, B, Bt, Wt, 0, "decision_f0_{}.png".format(tag))
    draw_decision_full(dec1, B, Bt, Wt, 1, "decision_f1_{}.png".format(tag))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Optimal pull/save strategy (Objective B)  -  budget B={}, "
                 "featured worth W={} pulls\nGenshin 5-star pity, classic 50/50 "
                 "(q=0.5), cost=1 pull/pull".format(Bt, Wt), fontsize=13)

    draw_decision(axes[0, 0], dec0, B,
                  "Decision map  f=0  (no guarantee)\nteal = PULL, gray = SAVE")
    draw_decision(axes[0, 1], dec1, B,
                  "Decision map  f=1  (next 5-star guaranteed featured)")
    draw_value(axes[1, 0], val0, B, "Value function V*(b,k)  for f=0")

    # threshold comparison
    ax = axes[1, 1]
    budgets = np.arange(1, B + 1)
    ax.plot(threshold(dec0), budgets, color="#e63946", lw=2, label="f=0 (no guarantee)")
    ax.plot(threshold(dec1), budgets, color="#1d3557", lw=2, label="f=1 (guaranteed)")
    ax.axvline(SOFT, color="#264653", ls="--", lw=1.2, label="soft pity (k=73)")
    ax.set_xlim(0, N - 1)
    ax.set_ylim(1, B)
    ax.set_xlabel("pity threshold k*(b)")
    ax.set_ylabel("budget b (pulls left)")
    ax.set_title("Threshold curves: pull iff pity >= k*(b)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "strategy_{}.png".format(tag)
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
