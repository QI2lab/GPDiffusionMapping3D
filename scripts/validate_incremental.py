"""Stage 1 gate: compare existing proposal arithmetic with interpolation."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions


def main():
    """Expose the regularization inconsistency before backend comparison."""
    inducing = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    observations = np.array([[.2, .3], [.8, .7]])
    covariance = functions.covMat(inducing, inducing, 2., 1.)
    cross = functions.covMat(inducing, observations, 2., 1.)
    regularized = covariance + .01 * covariance.mean() * np.eye(4)
    interpolation = cross.T @ np.linalg.inv(regularized)
    old = np.array([1., 1.1, .9, 1.05])
    change = .2
    # These are the two updates in diffusionPointSamplerSA_nb.
    proposed = old + covariance[:, 0] * change
    incremental = interpolation @ old + cross[0] * change
    direct = interpolation @ proposed
    # A conservative small dense matrix arithmetic bound, declared in advance.
    tolerance = (64 * np.finfo(np.float64).eps
                 * np.linalg.cond(regularized)
                 * max(1., np.max(np.abs(direct))))
    print(f"maximum field discrepancy: {np.max(np.abs(incremental-direct))}")
    print(f"float64 conditioning-based absolute tolerance: {tolerance}")
    np.testing.assert_allclose(incremental, direct, rtol=0, atol=tolerance)


if __name__ == "__main__":
    main()
