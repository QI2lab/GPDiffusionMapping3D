"""Summarize historical CPU/GPU archives using their original agreement rule."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.support.validate_3d import sampling_diagnostics, truth


def summarize(directory: Path, output: Path):
    """Record physical recovery, autocorrelation, and independent-chain errors."""
    results = {}
    means = []
    errors = []
    points = np.array(
        [
            (z, y, x)
            for z in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            for x in (-1.0, 0.0, 1.0)
        ]
    )
    for backend in ("cpu", "gpu"):
        with np.load(directory / f"{backend}.npz") as trace:
            projected = trace["projected"]
            posterior = trace["posterior"]
        mean = projected.mean(axis=0)
        batches = np.array_split(projected, 10)
        error = np.std(
            [b.mean(axis=0) for b in batches], axis=0, ddof=1
        ) / np.sqrt(10)
        means.append(mean)
        errors.append(error)
        results[backend] = {
            **sampling_diagnostics(projected),
            "iterations": len(projected),
            "center_mean": float(mean[13]),
            "map_recovery_rms": float(
                np.sqrt(
                    np.mean(
                        (
                            (
                                projected[np.argmax(posterior)]
                                - truth(points, True)
                            )
                            / truth(points, True)
                        )
                        ** 2
                    )
                )
            ),
            "mean_recovery_rms": float(
                np.sqrt(
                    np.mean(
                        ((mean - truth(points, True)) / truth(points, True))
                        ** 2
                    )
                )
            ),
        }
    difference = abs(means[0] - means[1])
    bound = t.ppf(1 - 0.01 / (2 * 27), df=9) * np.sqrt(
        errors[0] ** 2 + errors[1] ** 2
    )
    results["comparison"] = {
        "passed": bool(np.all(difference < bound)),
        "difference": difference.tolist(),
        "bound": bound.tolist(),
        "criterion": "Unchanged 99% family Student-t df=9 over 27 central positions",
    }
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.directory, args.output)
