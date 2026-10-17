"""Measure and physically validate the repaired 3D CPU defaults."""

import contextlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model
from scripts.validate_3d import prepared, truth


def measure(backend: str) -> dict:
    """Check recovery and densities, timing one warmed inference at a time."""
    data = prepared(feature=True)
    parameters = dict(nFineX=5, nFineY=5, nFineZ=5, covLambda=0.5, covL=1.5)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.analyze(
            data, backend=backend, nIter=1, seed=299,
            annealing_sweeps=1, adaptation_sweeps=1, burn_in=0, **parameters
        )
    warmup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        v, draws, posterior = model.analyze(
            data, backend=backend, seed=301,
            **parameters
        )
    seconds = time.perf_counter() - started
    coordinates = v.induCoordinates
    covariance = 0.25 * np.exp(
        -np.sum((coordinates[:, None] - coordinates[None]) ** 2, axis=2) / 4.5
    )
    covariance += 0.01 * covariance.mean() * np.eye(len(coordinates))
    condition = np.linalg.cond(covariance)
    weights = np.linalg.solve(
        covariance,
        0.25 * np.exp(
            -np.sum(
                (data.starts[:, None] - coordinates[None]) ** 2, axis=2
            ) / 4.5
        ).T,
    ).T
    prior_mean = np.mean(
        np.sum((data.ends - data.starts) ** 2, axis=1) / (6 * data.delta_t)
    )
    maximum_error = 0.0
    for index in (0, len(draws) // 2, len(draws) - 1):
        field = weights @ draws[index]
        assert np.all(field > 0) and np.all(draws[index] > 0)
        residual = draws[index] - prior_mean
        direct = stats.norm.logpdf(
            data.ends - data.starts,
            scale=np.sqrt(2 * field[:, None] * data.delta_t[:, None]),
        ).sum() - 0.5 * residual @ np.linalg.solve(covariance, residual)
        error = abs(direct - posterior[index])
        bound = 128 * np.finfo(float).eps * condition * max(1, abs(direct))
        assert error < bound
        maximum_error = max(maximum_error, float(error))
    points = np.array([
        (z, y, x) for z in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)
    ])
    central = np.linalg.solve(
        covariance,
        0.25 * np.exp(
            -np.sum((points[:, None] - coordinates[None]) ** 2, axis=2) / 4.5
        ).T,
    ).T @ draws[np.argmax(posterior)]
    expected = truth(points, True)
    rms = float(np.sqrt(np.mean(((central - expected) / expected) ** 2)))
    assert rms < 0.1
    return dict(
        backend=backend, dtype="float64", production_sweeps=len(draws),
        warmup_seconds=warmup_seconds, full_inference_seconds=seconds,
        map_rms=rms, full_density_max_error=maximum_error,
        device="CPU",
        schedule={key: v.sampling_metadata[key] for key in (
            "annealing_sweeps", "stabilization_sweeps",
            "discarded_fixed_width_sweeps", "production_sweeps",
            "proposal", "adaptation",
        )},
        qualification=(
            "Physical/default and timing check; not a convergence claim "
            "for 1000 draws"
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for backend in ("cpu",):
        result = measure(backend)
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
