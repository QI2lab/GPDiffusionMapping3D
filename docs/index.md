# GPDiffusionMapping

Infer a static spatial diffusion field from linked particle trajectories, with
one map per cell. Both 2D and timestamped 3D data are supported. Inference uses
CPU float64 with NumPy and Numba.

The method comes from Kumar et al.,
[DiffMAP-GP: Continuous 2D diffusion maps from particle trajectories without data binning using Gaussian processes](https://doi.org/10.1016/j.bpr.2024.100194),
implemented originally in
[LabPresse/GPDiffusionMapping](https://github.com/LabPresse/GPDiffusionMapping).
Original software: Copyright (c) 2024 Pressé Lab, under the
[MIT license](https://github.com/dpshepherd/GPDiffusionMapping/blob/main/LICENSE).
See [NOTICE](https://github.com/dpshepherd/GPDiffusionMapping/blob/main/NOTICE)
for attribution and dataset provenance.

## Get started

1. [Install](installation.md) the package with uv.
2. [Prepare linked tracks and run inference](workflow.md) for 2D or 3D cells.
3. [Read the maps](map-export.md) as CSV/Parquet and calibrated OME-TIFF.

Positions use microns, timestamps use seconds, and diffusion coefficients use
µm²/s. A directory contains one file per cell; a single combined file uses an
integer `cell_id` column. Batching is automatic.

## Method and development

The [API notes](api.md) describe computational entry points. The
[scientific validation records](reference/index.md) preserve the mathematical,
physical, sampling and performance evidence, including historical GPU work.
Current inference uses CPU float64; float32 is used only for TIFF export.

For development, see [Contributing](contributing.md) and the
[GitHub CI guide](github-ci.md).
