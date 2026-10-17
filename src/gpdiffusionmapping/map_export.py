# SPDX-License-Identifier: MIT
# Project attribution: see LICENSE and NOTICE.

"""Export physical diffusion maps as tables and calibrated OME-TIFF images."""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def export_map(
    directory: Path, name: str, cell_id: int, coordinates: np.ndarray,
    values: np.ndarray, *, dimensions: int, output_format: str,
) -> dict:
    """Write a float64 coordinate table and float32 YX/ZYX diffusion image.

    Coordinates and pixel-center origins are in microns; values are in
    micron^2/s. Pixel spacings come from the regular inference grid. Float32
    is an output encoding only; inference and tabular results remain float64.
    """
    if output_format not in {"csv", "parquet"}:
        raise ValueError("Map output format must be csv or parquet")
    coordinate_axes = "xy" if dimensions == 2 else "zyx"
    levels = {
        axis: np.unique(coordinates[:, index])
        for index, axis in enumerate(coordinate_axes)
    }
    spacing = {}
    for axis, positions in levels.items():
        if len(positions) < 2:
            raise ValueError(f"OME-TIFF needs at least two grid points on {axis}")
        step = float((positions[-1] - positions[0]) / (len(positions) - 1))
        if step <= 0 or not np.allclose(
            np.diff(positions), step, rtol=1e-10,
            atol=32 * np.finfo(float).eps * np.max(np.abs(positions)),
        ):
            raise ValueError("OME-TIFF requires a regular positive-spacing grid")
        spacing[axis] = step
    image_axes = "yx" if dimensions == 2 else "zyx"
    shape = tuple(len(levels[axis]) for axis in image_axes)
    if np.prod(shape) != len(values):
        raise ValueError("Map must contain a complete rectangular grid")
    indices = tuple(
        np.searchsorted(levels[axis], coordinates[:, coordinate_axes.index(axis)])
        for axis in image_axes
    )
    raster = np.empty(shape, dtype=np.float32)
    with np.errstate(over="ignore"):
        raster[indices] = values
    if not np.isfinite(raster).all():
        raise ValueError("Diffusion map cannot be represented as finite float32")
    table = pd.DataFrame({"cell_id": np.full(len(values), cell_id, dtype=np.int64)})
    for index, axis in enumerate(coordinate_axes):
        table[f"{axis}_um"] = coordinates[:, index]
    table["diffusion_um2_s"] = values
    target = directory / f"{name}.{output_format}"
    if output_format == "csv":
        table.to_csv(target, index=False)
    else:
        table.to_parquet(target, index=False)
    planes = shape[0] if dimensions == 3 else 1
    ome = {
        "axes": image_axes.upper(), "SignificantBits": 32,
        "Name": name, "Description": "MAP diffusion coefficient in µm²/s",
        "Plane": {},
    }
    for axis in coordinate_axes:
        key = axis.upper()
        ome[f"PhysicalSize{key}"] = spacing[axis]
        ome[f"PhysicalSize{key}Unit"] = "µm"
        ome["Plane"][f"Position{key}"] = (
            levels[axis].tolist() if axis == "z"
            else [float(levels[axis][0])] * planes
        )
        ome["Plane"][f"Position{key}Unit"] = ["µm"] * planes
    tifffile.imwrite(
        directory / f"{name}.ome.tiff", raster, ome=True, bigtiff=True,
        photometric="minisblack", metadata=ome,
        resolution=(1e4 / spacing["x"], 1e4 / spacing["y"]),
        resolutionunit="CENTIMETER",
    )
    return {
        "map_table": target.name, "map_image": f"{name}.ome.tiff",
        "image_axes": image_axes.upper(), "image_dtype": "float32",
        "pixel_size_um": spacing,
        "origin_um": {axis: float(points[0]) for axis, points in levels.items()},
    }
