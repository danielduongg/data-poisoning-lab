"""plots.py -- figures for the report (matplotlib, Agg backend)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402


def plot_sweep(rates, asr, clean_acc, path):
    x = [r * 100 for r in rates]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(x, [a * 100 for a in asr], "o-", color="#c0392b", lw=2, label="Attack success rate")
    ax1.set_xlabel("Poison fraction of training set (%)")
    ax1.set_ylabel("Attack success rate (%)", color="#c0392b")
    ax1.set_ylim(0, 102)
    ax2 = ax1.twinx()
    ax2.plot(x, [a * 100 for a in clean_acc], "s--", color="#2980b9", lw=2, label="Clean accuracy")
    ax2.set_ylabel("Clean test accuracy (%)", color="#2980b9")
    ax2.set_ylim(0, 102)
    ax1.set_title("Backdoor: a tiny poison fraction is enough (and stays stealthy)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mitigation(before, after, path):
    plt.figure(figsize=(4.6, 4.4))
    bars = plt.bar(["Poisoned\nmodel", "After defense\n(cleaned + retrain)"],
                   [before * 100, after * 100], color=["#c0392b", "#27ae60"], width=0.6)
    for b, v in zip(bars, [before, after]):
        plt.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v*100:.1f}%",
                 ha="center", fontsize=11)
    plt.ylabel("Attack success rate (%)")
    plt.ylim(0, 105)
    plt.title("Defense reduces the backdoor")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_clusters(X, cluster_labels, poison_truth, path, seed=0):
    """2-D TruncatedSVD scatter of target-labeled docs: predicted clusters vs. truth."""
    svd = TruncatedSVD(n_components=2, random_state=seed)
    pts = svd.fit_transform(X)
    truth = np.asarray(poison_truth, bool)
    plt.figure(figsize=(6, 4.6))
    plt.scatter(pts[~truth, 0], pts[~truth, 1], s=12, c="#2980b9", alpha=0.6, label="genuine target")
    plt.scatter(pts[truth, 0], pts[truth, 1], s=22, c="#c0392b", marker="^",
                alpha=0.85, label="poisoned (ground truth)")
    plt.xlabel("SVD-1")
    plt.ylabel("SVD-2")
    plt.title("Target-labeled training docs (poison separates out)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_attack_vs_defense(rates, asr, det_f1, path):
    x = [r * 100 for r in rates]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(x, [a * 100 for a in asr], "o-", color="#c0392b", lw=2,
             label="Attack success rate")
    plt.plot(x, [f * 100 for f in det_f1], "s--", color="#27ae60", lw=2,
             label="Defense detection F1")
    plt.xlabel("Poison fraction of training set (%)")
    plt.ylabel("Percent")
    plt.ylim(0, 102)
    plt.title("The defense is weakest where the attack is stealthiest")
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
