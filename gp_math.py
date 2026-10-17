"""Validated spatial GP and Gaussian transition mathematics."""

import numba as nb
import numpy as np


def expCooling(iteration, initialTemp, coolRate):
    """Evaluate the existing exponential annealing schedule."""
    return initialTemp * np.exp(-coolRate * iteration)


@nb.njit(cache=True)
def logNormpdf(diff, sigma):
    """Evaluate the original unnormalized scalar Gaussian log density."""
    return -np.log(np.abs(sigma)) - 0.5 * (diff / sigma) ** 2


@nb.njit(cache=True)
def indexShuffler(length):
    """Shuffle indices with the original sequential random draws."""
    indices = np.arange(length)
    for i in range(length):
        j = int(np.random.random() * (length - i)) + i
        indices[i], indices[j] = indices[j], indices[i]
    return indices


@nb.njit(cache=True)
def covMat(coordinates1, coordinates2, covLambda, covL):
    """Evaluate the existing isotropic spatial RBF kernel.

    Parameters
    ----------
    coordinates1, coordinates2 : numpy.ndarray
        XY or ZYX locations, with one location per row.
    covLambda : float
        Prior marginal standard deviation.
    covL : float
        Spatial correlation length in the same units as the coordinates.

    Returns
    -------
    numpy.ndarray
        Covariance between the two coordinate sets.
    """
    C = np.zeros((len(coordinates1), len(coordinates2)))
    for i in range(len(coordinates1)):
        for j in range(len(coordinates2)):
            dist_squared = 0.0
            for axis in range(coordinates1.shape[1]):
                dist_squared += (
                    coordinates1[i, axis] - coordinates2[j, axis]
                ) ** 2
            dist = np.sqrt(dist_squared)
            C[i, j] = (covLambda**2) * (
                np.exp(((-0.5) * ((dist) ** 2)) / (covL**2))
            )
    return C


def log_likelihood(diffusion, a, delta_t, dimensions=2):
    """Evaluate the full Gaussian transition density, including constants."""
    if bool(np.any(diffusion <= 0)):
        return -np.inf
    return -np.sum(
        dimensions / 2 * np.log(diffusion)
        + a / diffusion
        + dimensions / 2 * np.log(4 * np.pi * delta_t)
    )


def log_prior(diffusion, mean, inverse):
    """Evaluate the original unnormalized direct GP prior."""
    difference = diffusion - mean
    return -0.5 * (difference @ (inverse @ difference))


def likelihood_difference(old, new, a, dimensions=2):
    """Evaluate a proposal ratio without subtracting total densities."""
    if bool(np.any(new <= 0)):
        return -np.inf
    return np.sum(
        -dimensions / 2 * np.log(new / old) - a * (1 / new - 1 / old)
    )


def proposal_log_ratio(change, scale, old, new):
    """Return the original conditional proposal-density correction."""
    if old == 0 or new == 0:
        return -np.inf
    return float(
        logNormpdf(change, scale * new) - logNormpdf(change, scale * old)
    )


@nb.njit(cache=True)
def seed_sampler(seed):
    """Set the CPU sampler RNG for independently seeded inference runs."""
    np.random.seed(seed)
