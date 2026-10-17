"""File orchestration and scientific output, separate from sampling."""

import json
from pathlib import Path

import numpy as np

from data_reader import read_tracks
import model


def infer_file(
    path: str | Path, *, dimensions: int = 3, backend: str = "cpu",
    dtype: str = "float64", iterations: int = 1000, scale: float = 1,
    delta_t: float | None = None, particle_col: str = "particle #",
    z_col: str = "z", y_col: str = "y", x_col: str = "x",
    time_col: str = "timestamp", **model_options,
) -> tuple:
    """Infer a static field from linked CSV or Parquet tracks.

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
        Conversion of input lengths to nanometers.
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


def save_results(
    directory: str | Path, data, variables, samples: np.ndarray,
    posterior: np.ndarray,
) -> Path:
    """Save the diffusion figure, numerical samples and physical-unit metadata.

    Returns
    -------
    pathlib.Path
        Directory containing InferredMAP.png, inference.npz and metadata.json.
    """
    import matplotlib.pyplot as plt
    import plotting

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure = plotting.plot(variables, samples, posterior, data)
    figure.savefig(directory / "InferredMAP.png", dpi=150)
    plt.close(figure)
    map_values = variables.cInduFine.T @ (
        variables.cInduInduInv @ samples[np.argmax(posterior)]
    )
    np.savez_compressed(
        directory / "inference.npz", samples_nm2_s=samples,
        log_posterior=posterior,
        inducing_coordinates_nm=variables.induCoordinates,
        fine_coordinates_nm=variables.fineCoordinates,
        map_diffusion_nm2_s=map_values,
    )
    metadata = {
        "dimensions": variables.dimensions,
        "spatial_order": "ZYX" if variables.dimensions == 3 else "XY",
        "field": "static spatial diffusion",
        "backend": variables.backend,
        "dtype": str(samples.dtype),
        "units": {"position": "nm", "time": "s", "diffusion": "nm^2/s"},
        "figure_units": {"position": "um", "diffusion": "um^2/s"},
        "transitions": len(variables.dataCoordinates),
        "constant_diffusion_mle_nm2_s": float(variables.mle),
        "sampling": variables.sampling_metadata,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=lambda value: value.tolist())
        + "\n"
    )
    return directory
