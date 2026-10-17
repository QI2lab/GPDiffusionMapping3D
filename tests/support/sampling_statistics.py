"""Multiple-chain posterior diagnostics for scientific sampling validation."""

import numpy as np
from scipy import stats


def _split(chains):
    half = chains.shape[1] // 2
    return np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)


def _rank_normalize(values):
    shape = values.shape
    ranks = stats.rankdata(values.reshape(-1, shape[-1]), axis=0)
    return stats.norm.ppf((ranks - 0.375) / (len(ranks) + 0.25)).reshape(shape)


def _variance(values):
    n = values.shape[1]
    within = values.var(axis=1, ddof=1).mean(axis=0)
    total = (n - 1) / n * within + values.mean(axis=1).var(axis=0, ddof=1)
    return within, total


def _rhat(values):
    within, total = _variance(values)
    return np.sqrt(
        np.divide(
            total, within, out=np.full_like(total, np.inf), where=within > 0
        )
    )


def _ess(values):
    """Use between-chain variance and Geyer's initial monotone positive pairs."""
    count, n, dimensions = values.shape
    within, total = _variance(values)
    centered = values - values.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=2 * n, axis=1)
    covariance = np.fft.irfft(abs(spectrum) ** 2, n=2 * n, axis=1)[:, :n] / n
    average = covariance.mean(axis=0)
    result = np.zeros(dimensions)
    for axis in range(dimensions):
        if total[axis] <= 0:
            continue
        rho = 1 - (within[axis] - average[:, axis]) / total[axis]
        rho[0] = 1
        pairs = rho[: 2 * (n // 2)].reshape(-1, 2).sum(axis=1)
        stop = np.flatnonzero(pairs <= 0)
        if len(stop):
            pairs = pairs[: stop[0]]
        pairs = np.minimum.accumulate(pairs)
        tau = max(1.0, -1 + 2 * pairs.sum())
        result[axis] = min(count * n, count * n / tau)
    return result


def diagnose(chains):
    """Return rank/folded R-hat, bulk/tail ESS and mean Monte Carlo error.

    Parameters
    ----------
    chains : numpy.ndarray
        Independent chains with shape (chain, production draw, quantity).

    Returns
    -------
    dict
        Arrays for every monitored quantity; no diagnostic failure is skipped.
    """
    split = _split(np.asarray(chains))
    ranked = _rank_normalize(split)
    folded = _rank_normalize(abs(split - np.median(split, axis=(0, 1))))
    rhat = np.maximum(_rhat(ranked), _rhat(folded))
    bulk = _ess(ranked)
    low, high = np.quantile(split, [0.05, 0.95], axis=(0, 1))
    tail = np.minimum(
        _ess((split <= low).astype(float)), _ess((split >= high).astype(float))
    )
    _, variance = _variance(split)
    raw_ess = _ess(split)
    mcse = np.sqrt(
        np.divide(
            variance,
            raw_ess,
            out=np.full_like(variance, np.inf),
            where=raw_ess > 0,
        )
    )
    return {
        "rhat": rhat,
        "bulk_ess": bulk,
        "tail_ess": tail,
        "mean_mcse": mcse,
        "posterior_sd": np.sqrt(variance),
    }


def passes(diagnostics):
    """Apply the predeclared convergence requirements to every quantity."""
    return bool(
        np.all(diagnostics["rhat"] < 1.01)
        and np.all(diagnostics["bulk_ess"] >= 400)
        and np.all(diagnostics["tail_ess"] >= 400)
        and np.all(
            diagnostics["mean_mcse"] < 0.05 * diagnostics["posterior_sd"]
        )
    )
