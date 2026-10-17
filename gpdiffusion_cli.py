"""Thin Typer entry point for validated linked-track diffusion inference."""

import contextlib
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from mapping_io import infer_file, save_results


class Dimensions(str, Enum):
    """Supported spatial coordinate counts."""

    two = "2"
    three = "3"


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    tracks: Annotated[Path, typer.Argument(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Already linked tracks in .csv or .parquet format.",
    )],
    dimensions: Annotated[
        Dimensions, typer.Option(help="Spatial dimensions.")
    ] = Dimensions.three,
    particle_col: Annotated[
        str, typer.Option(help="Exact particle identity column.")
    ] = "particle #",
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
    """Infer a static isotropic diffusion field D(z,y,x) or D(x,y)."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(
            f"Inferring {dimensions.value}D diffusion on CPU (float64)"
        )
        with (output_dir / "sampling.log").open("w") as log:
            with contextlib.redirect_stdout(log):
                data, variables, samples, posterior = infer_file(
                    tracks, dimensions=int(dimensions.value),
                    iterations=iterations, scale=scale, delta_t=delta_t,
                    particle_col=particle_col, z_col=z_col, y_col=y_col,
                    x_col=x_col, time_col=time_col,
                )
        directory = save_results(
            output_dir, data, variables, samples, posterior
        )
    except (
        ValueError, TypeError, OSError, RuntimeError, ImportError
    ) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Saved diffusion map and samples to {directory}")


if __name__ == "__main__":
    app()
