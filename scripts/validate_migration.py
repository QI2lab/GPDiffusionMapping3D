"""Check unchanged 2D arithmetic against independent Brownian physics."""

import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions


def main():
    """Validate a fixed simulation without storing a frozen-output oracle."""
    rng = np.random.default_rng(20261016)
    diffusion, dt, count = 50000.0, 1 / 30, 20000
    steps = rng.normal(size=(count, 2)) * np.sqrt(2 * diffusion * dt)
    statistic = np.sum(steps**2) / (2 * diffusion * dt)
    bounds = stats.chi2.ppf([0.0005, 0.9995], 2 * count)
    assert bounds[0] < statistic < bounds[1]
    sigma = np.sqrt(2 * diffusion * dt)
    actual = np.sum(functions.logNormpdf(steps, sigma))
    expected = np.sum(stats.norm.logpdf(steps, scale=sigma))
    expected += count * np.log(2 * np.pi)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0)
    coords = np.array([[0.0, 0.0], [10.0, 20.0], [-5.0, 3.0]])
    kernel = functions.covMat(coords, coords, 2.0, 15.0)
    oracle = np.array(
        [
            [
                4
                * np.exp(
                    -sum((float(a) - float(b)) ** 2 for a, b in zip(p, q))
                    / 450
                )
                for q in coords
            ]
            for p in coords
        ]
    )
    np.testing.assert_allclose(kernel, oracle, rtol=2e-14, atol=0)
    print(
        json.dumps(
            {
                "versions": {
                    p: importlib.metadata.version(p)
                    for p in ("numpy", "numba", "scipy", "matplotlib")
                },
                "mle_nm2_s": float(
                    np.mean(np.sum(steps**2, axis=1)) / (4 * dt)
                ),
                "chi_square": float(statistic),
                "log_density_error": float(abs(actual - expected)),
                "kernel_max_error": float(np.max(abs(kernel - oracle))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
