# SPDX-License-Identifier: MIT
# Project attribution: see LICENSE and NOTICE.

"""Independent physical recovery checks through automatic cell batching."""

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gpdiffusionmapping.mapping_io import map_cells
from tests.support.validate_3d import simulate


CASES = ("csv", "parquet", "parquet-overrides", "csv-2d", "legacy-csv")


def validate_file_physics(directory: Path, backend: str, case: str, source=None):
    """Recover two distinct diffusion fields with overlapping particle IDs.

    Reuse seed44 Brownian increments for a controlled D=1 and D=2 experiment.
    Shift the second cell spatially so each map requires its own GP grid.
    Check Gaussian posteriors, analytic MLEs, chi-square uncertainty and the
    predeclared 10% central-region field recovery criterion independently.
    """
    legacy = case == "legacy-csv"
    overrides = case == "parquet-overrides"
    dimensions = 2 if legacy or case == "csv-2d" else 3
    ids, positions, times = (
        simulate(seed=44, irregular=not legacy) if source is None else source
    )
    if dimensions == 2:
        positions = positions[:, 1:]
    intervals = times[1::2] - times[::2]
    original_steps = positions[1::2] - positions[::2]
    frames, oracles = [], {}
    options = {}
    for index, diffusion in enumerate((1.0, 2.0)):
        cell_id = str(2**53 + 1 + 2 * index)
        starts = positions[::2] + 10 * index
        ends = starts + np.sqrt(diffusion) * original_steps
        coords = np.stack((starts, ends), axis=1).reshape(-1, dimensions)
        frame = pd.DataFrame({"particle_id": 2**53 + 1 + 2 * ids})
        axes = ("x", "y") if dimensions == 2 else ("z", "y", "x")
        for axis, values in zip(axes, coords.T):
            frame[axis] = values
        if legacy:
            frame = frame.rename(columns={
                "particle_id": "particle #", "x": "xPos", "y": "yPos",
            })
            options.update(
                particle_col="particle #", x_col="xPos", y_col="yPos",
                delta_t=0.005,
            )
        else:
            frame["timestamp"] = times
        frame["cell_id"] = cell_id
        frame["unused_column"] = "ignored"
        rng = np.random.default_rng(20261017)
        order = (
            (2 * rng.permutation(len(starts))[:, None] + (0, 1)).ravel()
            if legacy else rng.permutation(len(frame))
        )
        frame = frame.iloc[order]
        if overrides:
            frame = frame.rename(columns={
                "particle_id": "track_id", "cell_id": "cell_label",
                "z": "z_nm", "y": "y_nm", "x": "x_nm", "timestamp": "time_s",
            })
            frame[["z_nm", "y_nm", "x_nm"]] *= 1000
            options.update(
                particle_col="track_id", cell_col="cell_label", z_col="z_nm",
                y_col="y_nm", x_col="x_nm", time_col="time_s", scale=0.001,
            )
        frames.append(frame)
        oracles[cell_id] = diffusion, starts, ends - starts, index
    if legacy:
        path = directory / "cells"
        path.mkdir()
        for identity, frame, suffix in zip(oracles, frames, ("csv", "parquet")):
            frame = frame.drop(columns="cell_id")
            target = path / f"{identity}.{suffix}"
            if suffix == "csv":
                frame.to_csv(target, index=False)
            else:
                frame.to_parquet(target, index=False)
    else:
        path = directory / (
            "tracks.csv" if case in {"csv", "csv-2d"} else "tracks.parquet"
        )
        frame = pd.concat(frames).sample(frac=1, random_state=20261017)
        if path.suffix == ".csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False)
    model_options = dict(
        nFineX=5, nFineY=5, nFineZ=5, covLambda=0.5, covL=1.5, seed=601,
    )
    if dimensions == 2:
        model_options.update(nInduX=4, nInduY=4)
    results = []
    for output in map_cells(
        path, directory / "maps", dimensions=dimensions, backend=backend,
        iterations=100 if dimensions == 2 else 1000, **options, **model_options,
    ):
        metadata = json.loads((output / "metadata.json").read_text())
        diffusion, starts, steps, index = oracles.pop(metadata["cell_id"])
        count = len(steps)
        source_mle = np.mean(
            np.sum(steps**2, axis=1) / (2 * dimensions * intervals)
        )
        bound = 128 * np.finfo(float).eps * count
        np.testing.assert_allclose(
            metadata["constant_diffusion_mle_um2_s"], source_mle, rtol=bound
        )
        lower, upper = stats.chi2.ppf(
            [0.0005, 0.9995], dimensions * count
        ) / (dimensions * count)
        assert lower < source_mle / diffusion < upper
        with np.load(output / "inference.npz") as trace:
            coords = trace["inducing_coordinates_um"]
            posterior = trace["log_posterior"]
            inducing = trace["samples_um2_s"][np.argmax(posterior)]
        covariance = 0.25 * np.exp(
            -np.sum((coords[:, None] - coords[None])**2, axis=2) / 4.5
        )
        covariance += 0.01 * covariance.mean() * np.eye(len(coords))
        weights = 0.25 * np.exp(
            -np.sum((starts[:, None] - coords[None])**2, axis=2) / 4.5
        )
        field = weights @ np.linalg.solve(covariance, inducing)
        gaussian = stats.norm.logpdf(
            steps, scale=np.sqrt(2 * field[:, None] * intervals[:, None])
        ).sum()
        centered = inducing - source_mle
        expected = (
            gaussian - 0.5 * centered @ np.linalg.solve(covariance, centered)
        )
        posterior_bound = (
            128 * np.finfo(float).eps * (count + np.linalg.cond(covariance))
            * max(1, abs(expected))
        )
        np.testing.assert_allclose(
            posterior.max(), expected, rtol=0, atol=posterior_bound
        )
        points = np.array(
            list(product((-1., 0., 1.), repeat=dimensions))
        ) + 10 * index
        cross = 0.25 * np.exp(
            -np.sum((points[:, None] - coords[None])**2, axis=2) / 4.5
        )
        recovered = cross @ np.linalg.solve(covariance, inducing)
        rms = float(np.sqrt(np.mean(((recovered - diffusion) / diffusion)**2)))
        assert rms < 0.1
        results.append(dict(
            cell_id=metadata["cell_id"], diffusion=diffusion, map_rms=rms,
            posterior_error=float(abs(posterior.max() - expected)),
        ))
    # The two distinct physical fields must both be recovered independently.
    assert len(results) == 2 and not oracles
    (directory / "physical-result.json").write_text(
        json.dumps(dict(case=case, backend=backend, cells=results), indent=2)
        + "\n"
    )
    return results
