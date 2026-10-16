# SPDX-License-Identifier: MIT
# Project attribution: see LICENSE and NOTICE.

"""File orchestration and scientific output, separate from sampling."""

import contextlib
import json
from pathlib import Path

import numpy as np

from .data_reader import iter_cell_tracks, read_tracks
from .map_export import export_map
from . import model


def infer_file(
    path: str | Path, *, dimensions: int = 3, backend: str = "cpu",
    dtype: str = "float64", iterations: int = 1000, scale: float = 1,
    delta_t: float | None = None, particle_col: str = "particle_id",
    z_col: str = "z", y_col: str = "y", x_col: str = "x",
    time_col: str = "timestamp", **model_options,
) -> tuple:
    """Infer a static field from an already separated cell's linked tracks.

    Use map_cells for automatic grouping and per-cell output.

    Parameters
    ----------
    path : str or pathlib.Path
        Linked track table, selected by suffix.
    dimensions : {2, 3}, optional
        XY or ZYX spatial coordinates; timestamp is never a spatial axis.
    backend : {"cpu"}, optional
        Retained for existing CPU callers; other values are rejected.
    dtype : str, optional
        Validated float64 precision. Float32 failed scientific validation.
    iterations : int, optional
        Retained production sweeps, after the model's warmup schedule.
    scale : float, optional
        Conversion of input lengths to microns.
    delta_t : float or None, optional
        Required only for legacy 2D input without timestamps, in seconds.
    particle_col, z_col, y_col, x_col, time_col : str, optional
        Exact mapped column names.
    **model_options
        Computational model overrides, available independently of the CLI.

    Returns
    -------
    tuple
        Prepared observations, MAP state, production samples and log posterior.
    """
    if dtype != "float64":
        raise ValueError("Float32 failed scientific validation; use float64")
    data = read_tracks(
        path, dimensions=dimensions, scale=scale, delta_t=delta_t,
        particle_col=particle_col, z_col=z_col, y_col=y_col, x_col=x_col,
        time_col=time_col,
    )
    variables, samples, posterior = model.analyze(
        data, nIter=iterations, backend=backend, dtype=dtype, **model_options
    )
    return data, variables, samples, posterior


def map_cells(
    path: str | Path, output_dir: str | Path, *, dimensions: int = 3,
    cell_col: str = "cell_id", particle_col: str = "particle_id",
    z_col: str = "z", y_col: str = "y", x_col: str = "x",
    time_col: str = "timestamp", scale: float = 1,
    delta_t: float | None = None, iterations: int = 1000,
    backend: str = "cpu", dtype: str = "float64",
    output_format: str = "parquet", **model_options,
):
    """Infer and save one independent map per cell, yielding output directories.

    Parameters
    ----------
    path : str or pathlib.Path
        Combined CSV/Parquet file with cell identifiers, or a directory with
        one file per cell. Directory traversal is not recursive.
    output_dir : str or pathlib.Path
        Parent for zero-padded cell_<integer> subdirectories.
    output_format : {"csv", "parquet"}, optional
        Coordinate table format. A float32 OME-TIFF is always saved as well.
    dimensions, cell_col, particle_col, z_col, y_col, x_col, time_col, scale, delta_t
        Input settings accepted by iter_cell_tracks.
    iterations, backend, dtype, **model_options
        Per-cell inference settings accepted by model.analyze. Each cell gets
        a fresh GP grid, warmup and production chain; observations are not pooled.

    Yields
    ------
    pathlib.Path
        Cell directory containing the map, samples, metadata and sampling log.
    """
    if output_format not in {"csv", "parquet"}:
        raise ValueError("Map output format must be csv or parquet")
    for name, data in iter_cell_tracks(
        path, cell_col=cell_col, dimensions=dimensions,
        particle_col=particle_col, z_col=z_col, y_col=y_col, x_col=x_col,
        time_col=time_col, scale=scale, delta_t=delta_t,
    ):
        directory = Path(output_dir) / name
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "sampling.log").open("w") as log:
            with contextlib.redirect_stdout(log):
                variables, samples, posterior = model.analyze(
                    data, nIter=iterations, backend=backend, dtype=dtype,
                    **model_options,
                )
        yield save_results(
            directory, data, variables, samples, posterior,
            output_format=output_format, name=name,
        )


def save_results(
    directory: str | Path, data, variables, samples: np.ndarray,
    posterior: np.ndarray, *, output_format: str = "parquet", name: str | None = None,
) -> Path:
    """Save tabular and OME-TIFF maps, samples and physical-unit metadata.

    Returns
    -------
    pathlib.Path
        Directory containing calibrated maps, samples, figure and metadata.
    """
    import matplotlib.pyplot as plt
    from . import plotting

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = name or directory.name
    figure = plotting.plot(variables, samples, posterior, data)
    figure.savefig(directory / f"{name}.png", dpi=150)
    plt.close(figure)
    map_values = variables.cInduFine.T @ (
        variables.cInduInduInv @ samples[np.argmax(posterior)]
    )
    exported = export_map(
        directory, name, getattr(data, "cell_id", 0),
        variables.fineCoordinates, map_values,
        dimensions=variables.dimensions, output_format=output_format,
    )
    np.savez_compressed(
        directory / "inference.npz", samples_um2_s=samples,
        log_posterior=posterior,
        inducing_coordinates_um=variables.induCoordinates,
        fine_coordinates_um=variables.fineCoordinates,
        map_diffusion_um2_s=map_values,
    )
    metadata = {
        **exported,
        "dimensions": variables.dimensions,
        "spatial_order": "ZYX" if variables.dimensions == 3 else "XY",
        "field": "static spatial diffusion",
        "backend": variables.backend,
        "dtype": str(samples.dtype),
        "units": {"position": "um", "time": "s", "diffusion": "um^2/s"},
        "figure_units": {"position": "um", "diffusion": "um^2/s"},
        "transitions": len(variables.dataCoordinates),
        "constant_diffusion_mle_um2_s": float(variables.mle),
        "sampling": variables.sampling_metadata,
    }
    if hasattr(data, "cell_id"):
        metadata.update(cell_id=data.cell_id, source_file=data.source_file)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=lambda value: value.tolist())
        + "\n"
    )
    return directory
