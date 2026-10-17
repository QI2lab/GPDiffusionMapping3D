# SPDX-License-Identifier: MIT
# Project attribution: see LICENSE and NOTICE.

"""Thin Typer entry point for validated linked-track diffusion inference."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from .mapping_io import map_cells


class Dimensions(str, Enum):
    """Supported spatial coordinate counts."""

    two = "2"
    three = "3"


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    tracks: Annotated[Path, typer.Argument(
        exists=True, file_okay=True, dir_okay=True, readable=True,
        help="CSV/Parquet with cell_id, or a directory with one file per cell.",
    )],
    dimensions: Annotated[
        Dimensions, typer.Option(help="Spatial dimensions.")
    ] = Dimensions.three,
    particle_col: Annotated[
        str, typer.Option(help="Exact particle identity column.")
    ] = "particle_id",
    cell_col: Annotated[
        str, typer.Option(help="Exact cell identity column for single-file input.")
    ] = "cell_id",
    z_col: Annotated[
        str, typer.Option(help="Exact Z position column (3D only).")
    ] = "z",
    y_col: Annotated[
        str, typer.Option(help="Exact Y position column.")
    ] = "y",
    x_col: Annotated[
        str, typer.Option(help="Exact X position column.")
    ] = "x",
    time_col: Annotated[
        str, typer.Option(help="Exact timestamp column, numeric seconds.")
    ] = "timestamp",
    iterations: Annotated[
        int, typer.Option(min=1, help="Production sweeps after warmup.")
    ] = 1000,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for figure, samples and metadata.")
    ] = Path("output"),
    scale: Annotated[
        float, typer.Option(
            help="Multiply input positions by this value to obtain nm."
        )
    ] = 1.0,
    delta_t: Annotated[
        float | None, typer.Option(
            help="Seconds per frame; legacy 2D without timestamps only."
        )
    ] = None,
):
    """Infer a static diffusion field D(z,y,x) or D(x,y) for each cell."""
    try:
        typer.echo(
            f"Inferring {dimensions.value}D diffusion on CPU (float64)"
        )
        for directory in map_cells(
            tracks, output_dir, dimensions=int(dimensions.value),
            iterations=iterations, scale=scale, delta_t=delta_t,
            cell_col=cell_col, particle_col=particle_col, z_col=z_col,
            y_col=y_col, x_col=x_col, time_col=time_col,
        ):
            typer.echo(f"Saved cell map and samples to {directory}")
    except (
        ValueError, TypeError, OSError, RuntimeError, ImportError
    ) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    app()
