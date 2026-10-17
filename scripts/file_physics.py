"""Independent physical input checks shared by the separate hardware suites."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import functions as f
from mapping_io import infer_file
from scripts.validate_3d import simulate


CASES = ("csv", "parquet", "parquet-overrides", "legacy-csv")


def validate_file_physics(directory: Path, backend: str, case: str, source=None):
    """Recover diffusion after table serialization and physical conversion.

    Uses predeclared data seed44, sampler seeds CPU601/GPU701, 20000 linked
    pairs and the existing 10% central-region recovery threshold. The legacy
    XY uses regular .005-second observations; 3D uses actual irregular
    intervals. Identity values exceed 2**53, exposing lossy float conversion.
    """
    legacy = case == "legacy-csv"
    overrides = case == "parquet-overrides"
    dimensions = 2 if legacy else 3
    ids, positions, times = (
        simulate(seed=44, irregular=not legacy) if source is None else source
    )
    if legacy:
        positions = positions[:, 1:]
    steps = positions[1::2] - positions[::2]
    intervals = times[1::2] - times[::2]
    identities = 2**53 + 1 + 2 * ids
    options = {}
    if legacy:
        frame = pd.DataFrame({
            "particle #": identities, "xPos": positions[:, 0],
            "yPos": positions[:, 1],
        })
        options.update(x_col="xPos", y_col="yPos", delta_t=0.005)
    else:
        frame = pd.DataFrame({
            "particle #": identities, "z": positions[:, 0],
            "y": positions[:, 1], "x": positions[:, 2], "timestamp": times,
        })
    frame["unused_column"] = "ignored"
    rng = np.random.default_rng(20261017)
    if legacy:
        # Frame order within each identity is the only legacy time information.
        order = (2 * rng.permutation(len(steps))[:, None] + (0, 1)).ravel()
    else:
        order = rng.permutation(len(frame))
    frame = frame.iloc[order]
    if overrides:
        frame = frame.rename(columns={
            "particle #": "track_id", "z": "z_um", "y": "y_um",
            "x": "x_um", "timestamp": "time_s",
        })
        frame[["z_um", "y_um", "x_um"]] /= 1000
        options.update(
            particle_col="track_id", z_col="z_um", y_col="y_um",
            x_col="x_um", time_col="time_s", scale=1000,
        )
    path = directory / ("tracks.csv" if "csv" in case else "tracks.parquet")
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    model_options = dict(
        nFineX=5, nFineY=5, nFineZ=5, covLambda=0.5, covL=1.5,
        seed=601,
    )
    if legacy:
        model_options.update(nInduX=4, nInduY=4)
    data, v, samples, posterior = infer_file(
        path, dimensions=dimensions, backend=backend,
        iterations=100 if legacy else 1000, **options, **model_options,
    )
    count = len(steps)
    numeric_bound = 128 * np.finfo(float).eps * count
    source_mle = np.mean(
        np.sum(steps**2, axis=1) / (2 * dimensions * intervals)
    )
    np.testing.assert_allclose(v.mle, source_mle, rtol=numeric_bound)
    lower, upper = stats.chi2.ppf(
        [0.0005, 0.9995], dimensions * count
    ) / (dimensions * count)
    assert lower < v.mle < upper
    expected_density = stats.norm.logpdf(
        steps, scale=np.sqrt(2 * intervals[:, None])
    ).sum()
    actual_density = f.log_likelihood(
        np.ones(count), v.a, data.delta_t, dimensions
    )
    np.testing.assert_allclose(
        actual_density, expected_density, rtol=numeric_bound
    )
    if legacy:
        points = np.array([
            (x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)
        ])
    else:
        points = np.array([
            (z, y, x) for z in (-1., 0., 1.) for y in (-1., 0., 1.)
            for x in (-1., 0., 1.)
        ])
    covariance = 0.25 * np.exp(
        -np.sum(
            (v.induCoordinates[:, None] - v.induCoordinates[None]) ** 2,
            axis=2,
        ) / 4.5
    )
    covariance += 0.01 * covariance.mean() * np.eye(len(covariance))
    cross = 0.25 * np.exp(
        -np.sum((points[:, None] - v.induCoordinates[None]) ** 2, axis=2) / 4.5
    )
    recovered = cross @ np.linalg.solve(
        covariance, samples[np.argmax(posterior)]
    )
    rms = float(np.sqrt(np.mean((recovered - 1) ** 2)))
    assert rms < 0.1
    result = dict(
        backend=backend, case=case, dimensions=dimensions,
        mle=float(v.mle), map_rms=rms,
        gaussian_error=float(abs(actual_density - expected_density)),
    )
    (directory / "physical-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
