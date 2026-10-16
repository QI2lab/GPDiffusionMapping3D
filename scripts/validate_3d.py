"""Independent timestamped 3D physics and predeclared recovery protocol."""

import contextlib
import io
from pathlib import Path
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_reader
import functions as f
import model

SEEDS = (41, 42, 43)


def truth(points, feature=False):
    """Define an unbounded smooth positive 3D diffusion field in nm²/s."""
    return 1 + (
        0.4 * np.exp(-np.sum(points**2, axis=1) / 2)
        if feature
        else np.zeros(len(points))
    )


def simulate(count=20000, seed=41, feature=False, irregular=True, substeps=1):
    """Simulate independent linked ZYX pairs with known actual elapsed time."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(-3, 3, (count, 3))
    dt = (
        rng.uniform(0.002, 0.008, count)
        if irregular
        else np.full(count, 0.005)
    )
    end = start.copy()
    for _ in range(substeps):
        end += np.sqrt(
            2 * truth(end, feature)[:, None] * dt[:, None] / substeps
        ) * rng.normal(size=end.shape)
    positions = np.stack((start, end), axis=1).reshape(-1, 3)
    ids = np.repeat(np.arange(count), 2)
    times = np.stack((np.zeros(count), dt), axis=1).reshape(-1)
    return ids, positions, times


def prepared(count=20000, seed=41, feature=False, irregular=True, substeps=1):
    """Prepare physical transitions with the same path as user tracks."""
    return data_reader.prepare_tracks(
        *simulate(count, seed, feature, irregular, substeps)
    )


def physics(seed=41, irregular=True):
    """Use independent chi-square and Gaussian expectations in three axes."""
    ids, pos, times = simulate(seed=seed, irregular=irregular)
    steps = pos[1::2] - pos[::2]
    tau = times[1::2] - times[::2]
    normalized = steps / np.sqrt(2 * tau[:, None])
    n = len(steps)
    for component in normalized.T:
        assert abs(component.mean()) < 6 / np.sqrt(n)
        assert (
            stats.chi2.ppf(0.0005, n)
            < sum(component**2)
            < stats.chi2.ppf(0.9995, n)
        )
    total = np.sum(normalized**2)
    assert (
        stats.chi2.ppf(0.0005, 3 * n) < total < stats.chi2.ppf(0.9995, 3 * n)
    )
    data = data_reader.prepare_tracks(ids, pos, times)
    a = np.sum((data.ends - data.starts) ** 2, axis=1) / (4 * data.delta_t)
    expected = sum(
        stats.multivariate_normal.logpdf(step, cov=2 * t * np.eye(3))
        for step, t in zip(steps, tau)
    )
    np.testing.assert_allclose(
        f.log_likelihood(np.ones(n), a, data.delta_t, 3),
        expected,
        rtol=128 * np.finfo(float).eps * n,
    )
    mle = np.mean(np.sum(steps**2, axis=1) / (6 * tau))
    assert abs(mle - 1) < max(
        abs(stats.chi2.ppf(q, 3 * n) / (3 * n) - 1) for q in (0.0005, 0.9995)
    )
    return dict(
        msd_per_time=float(np.mean(np.sum(steps**2, axis=1) / tau)),
        expected=6.0,
        mle=float(mle),
    )


def recovery(
    backend="cpu",
    seed=41,
    feature=False,
    irregular=True,
    substeps=1,
    iterations=100,
    quiet=True,
    trace_path=None,
    data_seed=None,
    burn_in=0,
):
    """Recover a predeclared central cube with the original sampler schedule."""
    data = prepared(
        seed=seed if data_seed is None else data_seed,
        feature=feature,
        irregular=irregular,
        substeps=substeps,
    )
    if backend == "gpu":
        import cupy as cp

        cp.cuda.Device().synchronize()
    start = time.perf_counter()
    with (
        contextlib.redirect_stdout(io.StringIO())
        if quiet
        else contextlib.nullcontext()
    ):
        v, samples, p = model.analyze(
            data,
            nIter=iterations,
            backend=backend,
            seed=seed,
            burn_in=burn_in,
            nInduX=5,
            nInduY=5,
            nInduZ=5,
            nFineX=5,
            nFineY=5,
            nFineZ=5,
            covLambda=0.5,
            covL=1.5,
        )
    if backend == "gpu":
        cp.cuda.Device().synchronize()
    points = np.array(
        [
            (z, y, x)
            for z in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            for x in (-1.0, 0.0, 1.0)
        ]
    )
    weights = (
        f.covMat(points, v.induCoordinates, v.covLambda, v.covL)
        @ v.cInduInduInv
    )
    estimate = weights @ v.dIndu
    projected = samples @ weights.T
    if trace_path is not None:
        np.savez_compressed(
            trace_path,
            projected=projected,
            samples=samples,
            posterior=p,
            **v.sampling_metadata,
        )
    batches = np.array_split(projected, 10)
    mc_se = np.std(
        [batch.mean(axis=0) for batch in batches], axis=0, ddof=1
    ) / np.sqrt(10)
    error = float(
        np.sqrt(
            np.mean(
                ((estimate - truth(points, feature)) / truth(points, feature))
                ** 2
            )
        )
    )
    return dict(
        backend=backend,
        seed=seed,
        feature=feature,
        irregular=irregular,
        substeps=substeps,
        iterations=iterations,
        burn_in=burn_in,
        seconds=time.perf_counter() - start,
        recovery_rms=error,
        estimate=estimate.tolist(),
        posterior_mean=projected.mean(axis=0).tolist(),
        mc_se=mc_se.tolist(),
        sampling=sampling_diagnostics(projected),
        gpu_pool_bytes=cp.get_default_memory_pool().total_bytes()
        if backend == "gpu"
        else 0,
    )


def sampling_diagnostics(projected):
    """Report autocorrelation estimates without changing any scientific gate."""
    n = len(projected)
    centered = projected - projected.mean(axis=0)
    spectrum = np.fft.rfft(centered, n=2 * n, axis=0)
    covariance = np.fft.irfft(spectrum.conj() * spectrum, n=2 * n, axis=0)[
        :n
    ].real
    covariance /= np.arange(n, 0, -1)[:, None]
    rho = covariance / covariance[0]
    tau = []
    for axis in range(projected.shape[1]):
        pairs = rho[: 2 * (n // 2), axis].reshape(-1, 2).sum(axis=1)
        positive = np.flatnonzero(pairs <= 0)
        if len(positive):
            pairs = pairs[: positive[0]]
        pairs = np.minimum.accumulate(pairs)
        tau.append(max(1.0, -1 + 2 * pairs.sum()))
    return dict(
        min_estimated_ess=float(np.min(n / np.array(tau))),
        max_estimated_tau=float(max(tau)),
        maximum_lag1_correlation=float(rho[1].max()),
        maximum_split_mean_shift=float(
            abs(
                projected[: n // 2].mean(axis=0)
                - projected[n // 2 :].mean(axis=0)
            ).max()
        ),
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    result = recovery(args.backend, 41, True, iterations=args.iterations)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def logged_recovery(backend, seed, data_seed, iterations, directory):
    """Run one independent local diagnostic chain, retaining its trace and log."""
    import json

    directory = Path(directory)
    with (directory / f"{backend}.log").open("w", buffering=1) as stream:
        with contextlib.redirect_stdout(stream):
            result = recovery(
                backend,
                seed,
                True,
                iterations=iterations,
                data_seed=data_seed,
                quiet=False,
                trace_path=directory / f"{backend}.npz",
            )
    (directory / f"{backend}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
