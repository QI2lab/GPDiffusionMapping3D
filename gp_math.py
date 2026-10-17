"""Validated spatial GP and Gaussian transition mathematics."""

import numba as nb
import numpy as np

_proposal_reduction = None
_proposal_reduction32 = None


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



def covMat_gpu(coordinates1, coordinates2, covLambda, covL):
    """Evaluate the existing spatial RBF on CUDA using CuPy built-ins."""
    import cupy as cp

    squared = cp.zeros(
        (len(coordinates1), len(coordinates2)), dtype=coordinates1.dtype
    )
    for axis in range(coordinates1.shape[1]):
        squared += (
            coordinates1[:, axis, None] - coordinates2[None, :, axis]
        ) ** 2
    return covLambda**2 * cp.exp(-0.5 * squared / covL**2)



def log_likelihood(diffusion, a, delta_t, dimensions=2, xp=np):
    """Evaluate the full Gaussian transition density, including constants."""
    if diffusion.dtype == np.dtype("float32"):
        delta_t = xp.asarray(delta_t, dtype=diffusion.dtype)
    if bool(xp.any(diffusion <= 0)):
        return -xp.inf
    return -xp.sum(
        dimensions / 2 * xp.log(diffusion)
        + a / diffusion
        + dimensions / 2 * xp.log(4 * xp.pi * delta_t)
    )



def log_prior(diffusion, mean, inverse):
    """Evaluate the original unnormalized direct GP prior."""
    difference = diffusion - mean
    return -0.5 * (difference @ (inverse @ difference))



def likelihood_difference(old, new, a, dimensions=2, xp=np):
    """Evaluate a proposal ratio without subtracting total densities."""
    if bool(xp.any(new <= 0)):
        return -xp.inf
    return xp.sum(
        -dimensions / 2 * xp.log(new / old) - a * (1 / new - 1 / old)
    )



def proposal_log_ratio(change, scale, old, new):
    """Return the original conditional proposal-density correction."""
    if old == 0 or new == 0:
        return -np.inf
    return float(
        logNormpdf(change, scale * new) - logNormpdf(change, scale * old)
    )



def gpu_likelihood_difference(old, column, change, a, dimensions=2):
    """Fuse the measured proposal likelihood and positivity reduction."""
    import cupy as cp

    global _proposal_reduction, _proposal_reduction32
    if old.dtype == cp.float32:
        if _proposal_reduction32 is None:
            _proposal_reduction32 = cp.ReductionKernel(
                "float32 old, float32 column, float32 change, "
                "float32 a, float32 half_dimension",
                "float32 result",
                "old + column * change > 0 ? "
                "-half_dimension * logf((old + column * change) / old) "
                "-a * (1.0f / (old + column * change) - 1.0f / old) "
                ": -(1.0f/0.0f)",
                "a + b",
                "result = a",
                "0",
                "gpd_proposal_float32_spatial",
            )
        return _proposal_reduction32(old, column, change, a, dimensions / 2)
    if _proposal_reduction is None:
        _proposal_reduction = cp.ReductionKernel(
            "float64 old, float64 column, float64 change, "
            "float64 a, float64 half_dimension",
            "float64 result",
            "old + column * change > 0 ? "
            "-half_dimension * log((old + column * change) / old) "
            "-a * (1.0 / (old + column * change) - 1.0 / old) : -(1.0/0.0)",
            "a + b",
            "result = a",
            "0",
            "gpd_proposal_float64_spatial",
        )
    return _proposal_reduction(old, column, change, a, dimensions / 2)



@nb.njit(cache=True)
def seed_sampler(seed):
    """Set the CPU sampler RNG for independently seeded inference runs."""
    np.random.seed(seed)

