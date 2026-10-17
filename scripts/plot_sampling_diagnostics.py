"""Plot historical CPU/GPU trace archives; no CUDA execution is performed."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_traces(directory: Path, output: Path) -> None:
    """Plot central-field samples, running means, and posterior traces."""
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True, layout="constrained"
    )
    for backend, color in (("cpu", "tab:blue"), ("gpu", "tab:orange")):
        with np.load(directory / f"{backend}.npz") as trace:
            center = trace["projected"][:, 13]
            posterior = trace["posterior"]
        iteration = np.arange(1, len(center) + 1)
        axes[0].plot(
            iteration,
            center,
            color=color,
            alpha=0.6,
            lw=0.7,
            label=backend.upper(),
        )
        axes[1].plot(
            iteration,
            np.cumsum(center) / iteration,
            color=color,
            label=backend.upper(),
        )
        axes[2].plot(
            iteration,
            posterior,
            color=color,
            alpha=0.7,
            lw=0.7,
            label=backend.upper(),
        )
    axes[0].axhline(
        1.4, color="black", ls="--", label="simulation truth at ZYX=(0,0,0)"
    )
    axes[1].axhline(1.4, color="black", ls="--")
    axes[0].set_ylabel("D(0,0,0) (nm²/s)")
    axes[1].set_ylabel("Running mean D (nm²/s)")
    axes[2].set(ylabel="Log posterior", xlabel="Production iteration")
    axes[0].legend(loc="upper right")
    fig.suptitle(
        "Independent CPU/GPU chains on identical irregular-time 3D tracks"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_traces(args.directory, args.output)
