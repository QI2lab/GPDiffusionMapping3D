"""Staged 2D mathematics, simulated recovery, and synchronized benchmarks."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions as f
import model

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
    """Use fixed GP settings for all paired backend checks and simulations."""
    return {
        **model.PARAMETERS,
        "nInduX": 6,
        "nInduY": 6,
        "nFineX": 6,
        "nFineY": 6,
        "covLambda": 0.5,
        "covL": 1.2,
    }


def mathematical_checks(count=10000):
    """Check CUDA arithmetic against independent full-density mathematics."""
    import cupy as cp

    data = simulate(count)
    with contextlib.redirect_stdout(io.StringIO()):
        cpu = f.initialization(parameters(), data)
        gpu = f.initialization(parameters(), data, backend="gpu")
    condition = np.linalg.cond(cpu.cProposal)
    eps = np.finfo(float).eps
    relative = 128 * eps * condition
    errors = {}
    for key in (
        "cInduIndu",
        "cInduData",
        "cInduInduInv",
        "cDataIndu",
        "dIndu",
        "dData",
    ):
        a, b = getattr(cpu, key), cp.asnumpy(getattr(gpu, key))
        errors[key] = float(np.max(np.abs(a - b)))
        np.testing.assert_allclose(a, b, rtol=relative, atol=relative)
    full = stats.norm.logpdf(
        cpu.sampleCoordinates,
        loc=cpu.dataCoordinates,
        scale=np.sqrt(2 * cpu.dData[:, None] * data.deltaT),
    ).sum()
    ll = f.log_likelihood(cpu.dData, cpu.a, data.deltaT)
    np.testing.assert_allclose(ll, full, rtol=128 * eps * count)
    old = cpu.dIndu
    oldfield = cpu.dData
    ratios = []
    for k, change in ((0, 0.01), (17, -0.02), (35, 0.03)):
        new = old + cpu.cProposal[:, k] * change
        field = oldfield + cpu.cInduData[k] * change
        direct = cpu.cDataIndu @ new
        np.testing.assert_allclose(field, direct, rtol=relative, atol=relative)
        scalar = (
            stats.norm.logpdf(
                cpu.sampleCoordinates,
                loc=cpu.dataCoordinates,
                scale=np.sqrt(2 * field[:, None] * data.deltaT),
            ).sum()
            - full
        )
        ratio = f.likelihood_difference(oldfield, field, cpu.a)
        gpu_ratio = float(
            f.likelihood_difference(
                cp.asarray(oldfield), cp.asarray(field), gpu.a, xp=cp
            )
        )
        bound = 128 * eps * count * max(1, abs(full))
        assert abs(ratio - scalar) < bound
        assert abs(ratio - gpu_ratio) < bound
        fused = float(
            f.gpu_likelihood_difference(
                cp.asarray(oldfield),
                cp.asarray(cpu.cInduData[k]),
                change,
                gpu.a,
            )
        )
        assert abs(ratio - fused) < bound
        prior_delta = f.log_prior(
            new, cpu.priorMean, cpu.cInduInduInv
        ) - f.log_prior(old, cpu.priorMean, cpu.cInduInduInv)
        analytic_prior = (
            -change * (old - cpu.priorMean)[k]
            - 0.5 * change**2 * cpu.cProposal[k, k]
        )
        np.testing.assert_allclose(
            prior_delta, analytic_prior, rtol=relative, atol=relative
        )
        alpha = (cpu.cInduInduInv @ old)[k]
        correction = stats.norm.logpdf(
            change, scale=abs(0.3 * (alpha + change))
        ) - stats.norm.logpdf(change, scale=abs(0.3 * alpha))
        np.testing.assert_allclose(
            f.proposal_log_ratio(change, 0.3, alpha, alpha + change),
            correction,
            rtol=relative,
            atol=relative,
        )
        acceptance = ratio / 2 + prior_delta / 2 + correction
        gpu_acceptance = gpu_ratio / 2 + prior_delta / 2 + correction
        for log_u in (min(-0.1, acceptance - 1), min(-0.1, acceptance + 1)):
            if abs(acceptance - log_u) > bound:
                assert (acceptance > log_u) == (gpu_acceptance > log_u)
        ratios.append(abs(ratio - gpu_ratio))
    assert f.log_likelihood(np.array([0.0]), np.array([1.0]), 0.1) == -np.inf
    assert (
        f.likelihood_difference(
            cp.ones(2), cp.array([-1.0, 1.0]), cp.ones(2), xp=cp
        )
        == -cp.inf
    )
    cp.cuda.Device().synchronize()
    return dict(
        condition=condition,
        errors=errors,
        ratio_max_error=max(ratios),
        full_density_error=float(abs(ll - full)),
        relative_bound=relative,
    )


def inference(
    count, backend, seed, feature, substeps=1, iterations=100, data_seed=None
):
    """Measure the complete unchanged schedule and physical map recovery."""
    data = simulate(
        count, seed if data_seed is None else data_seed, feature, substeps
    )
    if backend == "gpu":
        import cupy as cp

        cp.cuda.Device().synchronize()
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
    if backend == "gpu":
        cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start
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
        mc_error=mc_error,
        estimate=estimate.tolist(),
        gpu_pool_bytes=cp.get_default_memory_pool().total_bytes()
        if backend == "gpu"
        else 0,
    )


def benchmark(count, backend, sweeps=10):
    """Separate initialization and warmed-up sequential sweep timing."""
    data = simulate(count)
    if backend == "gpu":
        import cupy as cp

        cp.cuda.Device().synchronize()
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        v = f.initialization(parameters(), data, backend=backend)
        if backend == "gpu":
            cp.cuda.Device().synchronize()
        init = time.perf_counter() - start
        v.epsilon = np.full(v.nIndu, 0.1)
        first = time.perf_counter()
        f.diffusionPointSampler(v, data)
        if backend == "gpu":
            cp.cuda.Device().synchronize()
        warmup = time.perf_counter() - first
        start = time.perf_counter()
        for _ in range(sweeps):
            f.diffusionPointSampler(v, data)
        if backend == "gpu":
            cp.cuda.Device().synchronize()
    return dict(
        count=count,
        backend=backend,
        initialization_s=init,
        first_sweep_s=warmup,
        sweep_s=(time.perf_counter() - start) / sweeps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["math", "benchmark", "inference"])
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--backend", default="gpu")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--feature", action="store_true")
    parser.add_argument("--substeps", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "math":
        result = mathematical_checks(args.count)
    elif args.mode == "benchmark":
        result = benchmark(args.count, args.backend)
    else:
        result = inference(
            args.count, args.backend, args.seed, args.feature, args.substeps
        )
    print(json.dumps(result, indent=2))


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


def independent_backend_checks(feature):
    """Compare three independent chains per backend on identical physical data."""
    from scipy.stats import t

    # Declared before execution, not selected using observed outcomes.
    cpu_seeds = (41, 42, 43)
    gpu_seeds = (141, 142, 143)
    cpu = [
        inference(10000, "cpu", seed, feature, data_seed=41)
        for seed in cpu_seeds
    ]
    gpu = [
        inference(10000, "gpu", seed, feature, data_seed=41)
        for seed in gpu_seeds
    ]
    assert all(row["recovery_rms"] < 0.1 for row in cpu + gpu)
    groups = [
        np.array([row["estimate"] for row in rows]) for rows in (cpu, gpu)
    ]
    se = np.sqrt(sum(group.var(axis=0, ddof=1) / 3 for group in groups))
    bound = t.ppf(1 - 0.01 / 18, df=2) * se
    difference = abs(groups[0].mean(axis=0) - groups[1].mean(axis=0))
    assert np.all(difference < bound)
    return dict(
        feature=feature,
        data_seed=41,
        cpu=cpu,
        gpu=gpu,
        differences=difference.tolist(),
        bounds=bound.tolist(),
    )
