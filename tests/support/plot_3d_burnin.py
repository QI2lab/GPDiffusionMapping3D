"""Plot historical CPU/GPU burn-in archives; no CUDA execution is performed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(directory: Path, report: Path, output: Path) -> None:
    """Show actual central-field traces, running means, and log posteriors."""
    results = json.loads(report.read_text())
    fig, axes = plt.subplots(
        3, 2, figsize=(13, 9), sharex=True, layout="constrained"
    )
    for column, (case, target) in enumerate(
        (("uniform", 1.0), ("smooth", 1.4))
    ):
        for backend, color in (("cpu", "tab:blue"), ("gpu", "tab:orange")):
            with np.load(directory / case / f"{backend}.npz") as trace:
                center = trace["projected"][:, 13]
                posterior = trace["posterior"]
            iteration = np.arange(1, len(center) + 1)
            axes[0, column].plot(
                iteration,
                center,
                color=color,
                alpha=0.65,
                lw=0.6,
                label=backend.upper(),
            )
            axes[1, column].plot(
                iteration,
                np.cumsum(center) / iteration,
                color=color,
                label=backend.upper(),
            )
            axes[2, column].plot(
                iteration, posterior, color=color, alpha=0.65, lw=0.6
            )
        for row in (0, 1):
            axes[row, column].axhline(
                target, color="black", ls="--", label="Simulation truth"
            )
        rhat = results["cases"][case]["rhat"]["maximum_rank_split_rhat"]
        axes[0, column].set_title(
            f"{case.title()}; max split R-hat over 27 sites: {rhat:.3f}"
        )
        axes[0, column].legend()
        axes[2, column].set_xlabel("Production sweep")
    axes[0, 0].set_ylabel("D(0,0,0) (nm²/s)")
    axes[1, 0].set_ylabel("Running mean D (nm²/s)")
    axes[2, 0].set_ylabel("Log posterior")
    fig.suptitle(
        "10,000 extra discarded temperature-1 sweeps; fixed proposal widths"
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
