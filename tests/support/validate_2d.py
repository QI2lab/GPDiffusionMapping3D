# SPDX-License-Identifier: MIT
# Project attribution: see LICENSE and NOTICE.

"""Independent 2D physical simulations, recovery and CPU benchmarks."""

import contextlib
import io
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpdiffusionmapping import functions as f
from gpdiffusionmapping import model
SEEDS = (41, 42, 43)
# Numerical bounds: float64 epsilon times accumulation/conditioning. Recovery:
# <=10% RMS in a predeclared densely sampled central square, across all seeds.
RECOVERY_TARGET = 0.10


def truth(points, feature=False):
    """Return a positive field defined throughout unbounded space, in nm²/s."""
    return 1 + (
        0.4 * np.exp(-np.sum(points**2, axis=1) / 2)
        if feature
        else np.zeros(len(points))
    )


def simulate(count=10000, seed=41, feature=False, substeps=1):
    """Simulate independent two-localization tracks using forward Euler."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(-3, 3, (count, 2))
    end = start.copy()
    dt = 0.005
    for _ in range(substeps):
        end += np.sqrt(
            2 * truth(end, feature)[:, None] * dt / substeps
        ) * rng.normal(size=end.shape)
    positions = np.stack((start, end), axis=1).reshape(-1, 2)
    return SimpleNamespace(
        trajectories=positions,
        trajectoriesIndex=np.repeat(np.arange(count), 2),
        nData=2 * count,
        deltaT=dt,
    )


def parameters():
    """Use fixed GP settings for physical simulations and inference."""
    return {
        **model.PARAMETERS,
        "nInduX": 6,
        "nInduY": 6,
        "nFineX": 6,
        "nFineY": 6,
        "covLambda": 0.5,
        "covL": 1.2,
    }




def inference(
    count, backend, seed, feature, substeps=1, iterations=100, data_seed=None
):
    """Measure the complete unchanged schedule and physical map recovery."""
    data = simulate(
        count, seed if data_seed is None else data_seed, feature, substeps
    )
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        v, samples, posterior = model.analyze(
            data,
            nIter=iterations,
            backend=backend,
            seed=seed,
            **{
                k: x
                for k, x in parameters().items()
                if k not in model.PARAMETERS
                or k
                in (
                    "nInduX",
                    "nInduY",
                    "nFineX",
                    "nFineY",
                    "covLambda",
                    "covL",
                )
            },
        )
    elapsed = time.perf_counter() - start
    # Independently rebuild the regularized GP and normalized Gaussian target.
    # In particular this checks accumulated incremental posterior updates, not
    # just recovery of a plausible MAP field.
    coordinates = v.induCoordinates
    covariance = v.covLambda**2 * np.exp(
        -np.sum((coordinates[:, None] - coordinates[None]) ** 2, axis=2)
        / (2 * v.covL**2)
    )
    covariance += 0.01 * covariance.mean() * np.eye(len(coordinates))
    cross = v.covLambda**2 * np.exp(
        -np.sum(
            (data.trajectories[::2, None] - coordinates[None]) ** 2, axis=2
        ) / (2 * v.covL**2)
    )
    independent_weights = np.linalg.solve(covariance, cross.T).T
    steps = data.trajectories[1::2] - data.trajectories[::2]
    prior_mean = np.mean(np.sum(steps**2, axis=1) / (4 * data.deltaT))
    maximum_density_error = 0.0
    for index in (0, len(samples) // 2, len(samples) - 1):
        field = independent_weights @ samples[index]
        assert np.all(field > 0) and np.all(samples[index] > 0)
        residual = samples[index] - prior_mean
        direct = stats.norm.logpdf(
            steps, scale=np.sqrt(2 * field[:, None] * data.deltaT)
        ).sum() - 0.5 * residual @ np.linalg.solve(covariance, residual)
        error = float(abs(direct - posterior[index]))
        bound = (
            128 * np.finfo(float).eps * np.linalg.cond(covariance)
            * max(1, abs(direct))
        )
        assert error < bound
        maximum_density_error = max(maximum_density_error, error)
    points = np.array(
        [(x, y) for x in np.linspace(-1, 1, 3) for y in np.linspace(-1, 1, 3)]
    )
    weights = (
        f.covMat(points, v.induCoordinates, v.covLambda, v.covL)
        @ v.cInduInduInv
    )
    estimate = weights @ v.dIndu
    error = float(
        np.sqrt(
            np.mean(
                ((estimate - truth(points, feature)) / truth(points, feature))
                ** 2
            )
        )
    )
    # Batch means give a conservative diagnostic of within-chain Monte Carlo noise.
    batches = np.array_split(samples @ weights.T, 5)
    mc_error = float(
        np.max(
            np.std([b.mean(axis=0) for b in batches], axis=0, ddof=1)
            / np.sqrt(5)
        )
    )
    return dict(
        count=count,
        backend=backend,
        seed=seed,
        feature=feature,
        substeps=substeps,
        iterations=iterations,
        seconds=elapsed,
        recovery_rms=error,
        full_density_max_error=maximum_density_error,
        mc_error=mc_error,
        estimate=estimate.tolist(),
    )


def benchmark(count, backend, sweeps=10):
    """Separate initialization and warmed-up sequential sweep timing."""
    data = simulate(count)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        v = f.initialization(parameters(), data, backend=backend)
        init = time.perf_counter() - start
        v.epsilon = np.full(v.nIndu, 0.1)
        first = time.perf_counter()
        f.diffusionPointSampler(v, data)
        warmup = time.perf_counter() - first
        start = time.perf_counter()
        for _ in range(sweeps):
            f.diffusionPointSampler(v, data)
    return dict(
        count=count,
        backend=backend,
        initialization_s=init,
        first_sweep_s=warmup,
        sweep_s=(time.perf_counter() - start) / sweeps,
    )


def brownian_checks(seed=41, count=20000):
    """Check component physics and the analytic constant-field MLE."""
    from scipy.optimize import minimize_scalar

    data = simulate(count, seed)
    steps = data.trajectories[1::2] - data.trajectories[::2]
    normalized = steps / np.sqrt(2 * data.deltaT)
    # 99.9% chi-square intervals per component; a 6-sigma mean interval.
    for component in normalized.T:
        assert abs(component.mean()) < 6 / np.sqrt(count)
        assert (
            stats.chi2.ppf(0.0005, count)
            < np.sum(component**2)
            < stats.chi2.ppf(0.9995, count)
        )
    total = np.sum(normalized**2)
    assert (
        stats.chi2.ppf(0.0005, 2 * count)
        < total
        < stats.chi2.ppf(0.9995, 2 * count)
    )
    analytic = np.sum(steps**2) / (4 * count * data.deltaT)

    def objective(d):
        return -stats.norm.logpdf(
            steps, scale=np.sqrt(2 * d * data.deltaT)
        ).sum()

    optimum = minimize_scalar(
        objective,
        bounds=(0.5, 1.5),
        method="bounded",
        options={"xatol": 1e-10},
    )
    # Bounded scalar optimization has approximately sqrt(epsilon) resolution.
    np.testing.assert_allclose(
        optimum.x, analytic, rtol=8 * np.sqrt(np.finfo(float).eps)
    )
    return dict(
        mean=steps.mean(axis=0).tolist(),
        msd=float(np.mean(np.sum(steps**2, axis=1))),
        expected_msd=4 * data.deltaT,
        mle=analytic,
    )
