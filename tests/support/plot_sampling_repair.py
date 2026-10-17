# SPDX-License-Identifier: MIT
# Project attribution: see LICENSE and NOTICE.

"""Plot historical CPU/GPU repair archives; no CUDA execution is performed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


def plot(directory: Path, report: Path, output: Path):
    """Show the central field and mixing diagnostics without thinning inference."""
    results = json.loads(report.read_text())
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for column, (case, target) in enumerate(
        (("uniform", 1.0), ("smooth", 1.4))
    ):
        arrays = []
        for backend in ("cpu", "gpu"):
            for index in range(4):
                with np.load(
                    directory / case / f"{backend}-{index}.npz"
                ) as trace:
                    arrays.append(trace["projected"][:, 13])
        ranks = rankdata(np.array(arrays).ravel()).reshape(8, -1)
        for line, center in enumerate(arrays):
            color = (
                plt.cm.Blues(0.45 + 0.15 * (line % 4))
                if line < 4
                else plt.cm.Oranges(0.45 + 0.15 * (line % 4))
            )
            label = f"{'CPU' if line < 4 else 'GPU'} {line % 4 + 1}"
            iteration = np.arange(1, len(center) + 1)
            # Display every tenth draw only; all draws enter statistics and means.
            axes[0, column].plot(
                iteration[::10],
                center[::10],
                color=color,
                alpha=0.5,
                lw=0.5,
                label=label,
            )
            axes[1, column].plot(
                iteration, np.cumsum(center) / iteration, color=color, lw=0.8
            )
            bins = np.linspace(0.5, ranks.size + 0.5, 21)
            histogram, _ = np.histogram(ranks[line], bins=bins)
            axes[2, column].stairs(histogram, bins, color=color, lw=1)
        axes[0, column].axhline(target, color="black", ls="--", lw=1)
        axes[1, column].axhline(target, color="black", ls="--", lw=1)
        axes[2, column].axhline(
            len(arrays[0]) / 20, color="black", ls="--", lw=1
        )
        rhat = max(
            results["cases"][case][backend]["checkpoints"][-1]["max_rhat"]
            for backend in ("cpu", "gpu")
        )
        axes[0, column].set_title(
            f"{case.title()}; max R-hat over all quantities {rhat:.4f}"
        )
        axes[0, column].legend(ncol=2, fontsize=7)
        axes[0, column].set_xlabel(
            "Production sweep (displayed every 10th draw)"
        )
        axes[1, column].set_xlabel("Production sweep (all draws)")
        axes[2, column].set_xlabel("Pooled rank at the central field point")
    axes[0, 0].set_ylabel("D(0,0,0) (µm²/s)")
    axes[1, 0].set_ylabel("Running mean D (µm²/s)")
    axes[2, 0].set_ylabel("Draws per rank bin, per chain")
    fig.suptitle(
        "Four independent CPU and four CUDA chains per field; fixed blocked MH proposals"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.directory, args.report, args.output)
