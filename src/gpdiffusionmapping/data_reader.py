# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Pressé Lab
# Derived from LabPresse/GPDiffusionMapping; see LICENSE and NOTICE.

import numpy as np
import copy
from types import SimpleNamespace

# CAPITALIZED COMMENTS FOR CLARITY

# THIS IS THE OBJECT OF DATA
DATA = {
    # NEED TO BE LOADED IN
    "trajectories": None,  # Coordinates of the data
    "trajectoriesIndex": None,  # Index of trajectory number
    "deltaT": None,  # Time passed in between each frame of data
    # INITIALIZED INDEPENDENTLY BASED ON PARAMETERS ABOVE
    "nData": None,  # Number of data points (set to len(nData))
    "nTrajectories": None,  # Number of trajectories (set to len(np.unique(data.trajectoriesIndex)))
}


def data_reader(path, scale=1, deltaT=1 / 30):
    """
    Read data from a CSV file and prepare it for further processing.

    Args:
        path (str): Path to the CSV file.
        scale (float): Scaling factor to transform data to nm. Defaults to 1.
        deltaT (float): Time interval between each frame of data. Defaults to 1/30.

    Returns:
        types.SimpleNamespace: An object containing trajectory data and related information.
    """

    # Read the CSV file, considering the header
    data = np.genfromtxt(path, delimiter=", ", skip_header=1)

    # Separate columns into individual arrays
    dataVectIndex = data[:, 0]
    dataVect = data[:, 1:]

    # Localization adjustment to nanometers
    dataVect = dataVect[::] * scale
    dataVectIndex = dataVectIndex[::]

    # Put time step manually as unavailable from data file
    deltaT = deltaT

    # Return dataVect, dataVectIndex, deltaT
    data = SimpleNamespace(**copy.deepcopy(DATA))
    data.trajectoriesIndex = dataVectIndex
    data.trajectories = dataVect
    data.deltaT = deltaT
    data.nData = len(data.trajectoriesIndex)
    data.nTrajectories = len(np.unique(data.trajectoriesIndex))

    return data


def prepare_tracks(particles, positions, timestamps, scale=1):
    """Prepare sorted within-particle 2D or ZYX 3D transitions.

    Parameters
    ----------
    particles : array-like
        Already linked identities, retained without floating-point conversion.
    positions : array-like
        XY columns for 2D or ZYX columns for 3D, in input length units.
    timestamps : array-like
        Numeric observation times in seconds.
    scale : float
        Positive conversion from input length units to nanometers.

    Returns
    -------
    types.SimpleNamespace
        Sorted tracks, consecutive start/end positions and positive elapsed times.
    """
    coordinates = np.asarray(positions, dtype=np.float64)
    raw_times = np.asarray(timestamps)
    if np.issubdtype(raw_times.dtype, np.datetime64) or np.issubdtype(
        raw_times.dtype, np.timedelta64
    ):
        raise ValueError(
            "Timestamps must be numeric seconds, not datetime values"
        )
    times = np.asarray(raw_times, dtype=np.float64)
    identities = np.asarray(particles, dtype=object)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise ValueError("Positions must have two XY or three ZYX columns")
    if (
        times.ndim != 1
        or identities.ndim != 1
        or len(times) != len(coordinates)
        or len(identities) != len(times)
    ):
        raise ValueError("Particle, position and timestamp rows must agree")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    coordinates = coordinates * scale
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(times)):
        raise ValueError("Positions and numeric timestamps must be finite")
    groups = {}
    for i, identity in enumerate(identities):
        if (
            identity is None
            or (isinstance(identity, str) and not identity)
            or (
                isinstance(identity, (float, np.floating))
                and not np.isfinite(identity)
            )
        ):
            raise ValueError("Particle identities must not be missing")
        groups.setdefault(identity, []).append(i)
    order = []
    starts = []
    ends = []
    intervals = []
    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        indices = indices[np.argsort(times[indices], kind="stable")]
        tau = np.diff(times[indices])
        if not np.all(np.isfinite(tau)) or np.any(tau <= 0):
            raise ValueError(
                "Each particle must have strictly increasing distinct timestamps"
            )
        order.extend(indices)
        if len(indices) > 1:
            starts.append(coordinates[indices[:-1]])
            ends.append(coordinates[indices[1:]])
            intervals.append(tau)
    if not intervals:
        raise ValueError(
            "Tracks must contain at least one positive-time transition"
        )
    return SimpleNamespace(
        trajectories=coordinates[order],
        trajectoriesIndex=identities[order],
        timestamps=times[order],
        nData=len(times),
        nTrajectories=len(groups),
        dimensions=coordinates.shape[1],
        starts=np.concatenate(starts),
        ends=np.concatenate(ends),
        delta_t=np.concatenate(intervals),
    )


def data_reader_3d(path, scale=1):
    """Read already-linked particle_id, z, y, x, timestamp CSV observations."""
    import csv
    from pathlib import Path

    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = ("particle_id", "z", "y", "x", "timestamp")
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError(f"3D CSV requires columns {required}")
        rows = list(reader)
    try:
        return prepare_tracks(
            [row["particle_id"] for row in rows],
            [[float(row[c]) for c in ("z", "y", "x")] for row in rows],
            [float(row["timestamp"]) for row in rows],
            scale=scale,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid timestamped 3D tracks: {error}") from error


def read_track_table(path, particle_col="particle_id", cell_col=None):
    """Read CSV or Parquet while preserving linked particle identities.

    Parameters
    ----------
    path : str or pathlib.Path
        Input table with a CSV or Parquet suffix.
    particle_col : str, optional
        Exact identity column. CSV identities are read as strings to avoid
        rounding large integer identifiers through floating point.
    cell_col : str or None, optional
        Preserve this CSV cell identifier as a string as well.

    Returns
    -------
    pandas.DataFrame
        Input columns with their original names and numerical values.
    """
    from pathlib import Path
    import pandas as pd

    path = Path(path)
    match path.suffix.lower():
        case ".csv":
            identities = {particle_col: "string"}
            if cell_col is not None:
                identities[cell_col] = "string"
            return pd.read_csv(
                path, dtype=identities, skipinitialspace=True
            )
        case ".parquet":
            return pd.read_parquet(path, engine="pyarrow")
        case _:
            raise ValueError("Track input must have a .csv or .parquet suffix")


def prepare_table_tracks(
    table, *, dimensions=3, particle_col="particle_id", z_col="z", y_col="y",
    x_col="x", time_col="timestamp", scale=1, delta_t=None,
):
    """Map table columns and prepare physical within-particle transitions.

    Parameters
    ----------
    table : pandas.DataFrame
        Already linked observations. Unrelated columns are ignored.
    dimensions : {2, 3}, optional
        Explicit XY or ZYX coordinate order.
    particle_col, z_col, y_col, x_col, time_col : str, optional
        Exact names of the corresponding columns.
    scale : float, optional
        Positive conversion from input position units to nanometers.
    delta_t : float or None, optional
        Explicit positive elapsed seconds for legacy 2D input without times.

    Returns
    -------
    types.SimpleNamespace
        Validated sorted tracks, transitions and actual elapsed seconds.
    """
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    coordinates = (x_col, y_col) if dimensions == 2 else (z_col, y_col, x_col)
    timestamped = time_col in table.columns
    required = (
        particle_col, *coordinates, *((time_col,) if timestamped else ())
    )
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing mapped columns: {', '.join(missing)}")
    if len(set(required)) != len(required):
        raise ValueError(
            "Mapped identity, position and time columns must be distinct"
        )
    if table[particle_col].isna().any():
        raise ValueError("Particle identities must not be missing")
    identities = table[particle_col].to_numpy(dtype=object)
    if timestamped:
        if delta_t is not None:
            raise ValueError(
                "delta_t cannot override an existing timestamp column"
            )
        times = table[time_col].to_numpy()
    else:
        if dimensions == 3:
            raise ValueError(f"3D input requires timestamp column {time_col!r}")
        if delta_t is None or not np.isfinite(delta_t) or delta_t <= 0:
            raise ValueError(
                "Legacy 2D input requires explicit positive delta_t seconds"
            )
        # In legacy input, row order within each linked particle is frame order.
        counts = {}
        times = np.empty(len(identities), dtype=np.float64)
        for index, identity in enumerate(identities):
            frame = counts.get(identity, 0)
            times[index] = frame * delta_t
            counts[identity] = frame + 1
    return prepare_tracks(
        identities, table[list(coordinates)].to_numpy(), times, scale=scale
    )


def read_tracks(path, **column_options):
    """Read either table format and reuse the same physical track preparation.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV or Parquet linked tracks.
    **column_options
        Exact column mappings, dimensions, scale and optional legacy delta_t
        accepted by prepare_table_tracks.

    Returns
    -------
    types.SimpleNamespace
        Prepared within-particle transitions in nanometers and seconds.
    """
    table = read_track_table(
        path, particle_col=column_options.get("particle_col", "particle_id")
    )
    return prepare_table_tracks(table, **column_options)


def iter_cell_tracks(
    path, *, cell_col="cell_id", dimensions=3, particle_col="particle_id",
    z_col="z", y_col="y", x_col="x", time_col="timestamp", scale=1,
    delta_t=None,
):
    """Yield separate physical tracks and safe output names for each cell.

    Parameters
    ----------
    path : str or pathlib.Path
        A CSV/Parquet file requiring a cell identifier column, or a directory
        whose immediate CSV/Parquet files each represent one cell.
    cell_col : str, optional
        Cell identifier column for a combined file. Directory cell names come
        from filenames; an optional cell column must identify only one cell.
    dimensions, particle_col, z_col, y_col, x_col, time_col, scale, delta_t
        Spatial dimensions, exact column mappings and physical conversions
        accepted by prepare_table_tracks.

    Yields
    ------
    tuple[str, types.SimpleNamespace]
        Output subdirectory name and independently prepared cell observations,
        including source_file and cell_id provenance.
    """
    from pathlib import Path
    from urllib.parse import quote

    path = Path(path)
    directory = path.is_dir()
    files = (
        sorted(p for p in path.iterdir()
               if p.is_file() and p.suffix.lower() in {".csv", ".parquet"})
        if directory else [path]
    )
    if not files:
        raise ValueError("Input directory contains no CSV or Parquet files")
    if cell_col in {particle_col, z_col, y_col, x_col, time_col}:
        raise ValueError("Cell column must be distinct from track columns")
    if directory and len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError("Each cell file must have a unique filename stem")
    names = set()
    for source in files:
        table = read_track_table(source, particle_col, cell_col)
        if table.empty:
            raise ValueError(f"No observations in {source}")
        if cell_col in table:
            values = table[cell_col]
            if values.isna().any() or values.astype(str).str.strip().eq("").any():
                raise ValueError(f"Every observation needs a {cell_col} in {source}")
        elif not directory:
            raise ValueError(f"Single-file input requires column {cell_col!r}")
        if directory:
            if cell_col in table and table[cell_col].nunique() != 1:
                raise ValueError(f"Each directory file must contain one cell: {source}")
            cells = [(source.stem, table)]
        else:
            cells = table.groupby(cell_col, sort=False, observed=True)
        for identity, rows in cells:
            identity = str(identity)
            name = "cell_" + quote(identity, safe="")
            if name.casefold() in names:
                raise ValueError(f"Cell output names collide for {identity!r}")
            names.add(name.casefold())
            data = prepare_table_tracks(
                rows, dimensions=dimensions, particle_col=particle_col,
                z_col=z_col, y_col=y_col, x_col=x_col, time_col=time_col,
                scale=scale, delta_t=delta_t,
            )
            data.cell_id = identity
            data.source_file = str(source.resolve())
            yield name, data
