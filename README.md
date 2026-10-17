# GPDiffusionMapping

Gaussian-process diffusion mapping from already linked particle trajectories.
Inference uses NumPy/Numba on CPU in float64, estimating a static isotropic
field in XY or ZYX coordinates. Synthetic and experimental data accompanying
the [paper](https://www.biorxiv.org/content/10.1101/2024.02.27.582378v1) are in
`data/`.

## Installation

Clone or download this repository, then use Python 3.12 and uv from the checkout:

```bash
uv sync
```

For development and the scientific tests:

```bash
uv sync --group dev
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
```

The suite uses only `unit` and `integration` scientific categories; select
either with `-m unit` or `-m integration`. Shared deterministic inputs and
independent numerical references are fixtures; MCMC states remain separate.
See [the test audit](docs/test-fixture-audit.txt).

`pyproject.toml` specifies dependencies. `uv.lock` is local and ignored; fresh
checkouts resolve their own environment. No GPU or CUDA installation is needed.

## Usage

The `gpdiffusion-map` Typer command reads `.csv` or `.parquet` containing
already linked tracks. Timestamps are numeric seconds and positions default to
nanometers. Each transition uses its actual positive elapsed time after sorting
within its particle. Timestamp is not a fourth spatial coordinate.

| Role | Default column | Override |
| --- | --- | --- |
| Particle identity | `particle #` | `--particle-col` |
| Z position (3D) | `z` | `--z-col` |
| Y position | `y` | `--y-col` |
| X position | `x` | `--x-col` |
| Time in seconds | `timestamp` | `--time-col` |

```bash
uv run --no-sync gpdiffusion-map tracks.csv
uv run --no-sync gpdiffusion-map tracks.parquet
uv run --no-sync gpdiffusion-map tracks.parquet --particle-col track_id --z-col z_um --y-col y_um --x-col x_um --time-col time_s --scale 1000
uv run --no-sync gpdiffusion-map legacy.csv --dimensions 2 --x-col xPos --y-col yPos --delta-t 0.03333333333333333
uv run --no-sync python -m gpdiffusion_cli tracks.csv
uv run --no-sync gpdiffusion-map --help
```

The scale example converts micrometers to nanometers. Column overrides are
exact and apply to both formats. Unrelated columns are ignored and particle
identities retain their precision. Three-dimensional input requires timestamps.
Legacy XY input without timestamps requires an explicit positive `--delta-t`
and uses row order within each linked particle. `--delta-t` is rejected when
timestamps are present.

Inference always uses CPU float64. Defaults are `--dimensions 3`,
`--iterations 1000`, `--output-dir output` and `--scale 1`.

Outputs are `InferredMAP.png`, `inference.npz`, `metadata.json` and
`sampling.log`. The figure uses µm and µm²/s; numerical coordinates and diffusion
samples use nm and nm²/s. The 3D figure is an XY slice at the nearest central Z.
NPZ contains every production inducing sample, unnormalized log posterior,
inducing/fine coordinates and the MAP field on the fine grid. Metadata records
coordinate order, units, CPU execution and the sampling schedule.

Reading, inference and output are callable separately through
`data_reader.read_tracks`, `model.analyze`, `mapping_io.infer_file` and
`mapping_io.save_results`. Model parameter overrides remain available through
the computational functions; see `model.PARAMETERS`. For example:

```python
import data_reader
import model

data = data_reader.read_tracks("tracks.parquet", dimensions=3)
variables, samples, posterior = model.analyze(data, seed=41)
```

Existing `backend="cpu"` and `dtype="float64"` computational API arguments
remain accepted. Other backends and precisions raise errors.

## Sampling and scientific validation

The original 2D sampler is preserved. Three-dimensional inference defaults to
eight-coordinate Fisher-preconditioned block Metropolis proposals, with 2,000
annealing sweeps, 4,000 temperature-one adaptation sweeps and 1,000 additional
discarded sweeps. Proposal geometry and widths stay fixed in production.
The exact Gaussian likelihood, direct GP prior and positivity constraints are
preserved. `burn_in` overrides the additional discarded sweeps. The historical
3D reference remains available with `proposal="relative", adaptation="legacy"`.

Production defaults to 1,000 sweeps. The convergence validation uses four
independently initialized chains with 20,000 production sweeps each, checking
all 125 inducing values, 125 inverse coefficients, 27 field points and log
posterior. A short single run does not establish convergence. Use independent
seeds and varied positive initial fields through the computational API; the
protocol is in `scripts/run_sampling_repair.py`.

The CPU suite checks independent Gaussian densities, GP mathematics,
Metropolis acceptance, posterior quadrature, Brownian physics, uniform/smooth
field recovery, irregular timestamps, full-chain convergence and physical
inference through CSV/Parquet with column overrides and unit conversion.
The latest local run passed **57 scientific tests** with zero skips: 23 unit
and 34 integration cases. See [the test audit results](docs/test-fixture-audit-results.json)
and [the CPU optimization results](docs/cpu-optimized-scientific-results.json).
CPU proposals reuse exact Gaussian likelihood terms and buffers, with an exact
incremental prior calculation for coefficient moves. The model and sampling
schedule are unchanged. See [the optimization report](docs/cpu-optimization.txt)
for measured performance and independent numerical checks. Isolated warmed
full-inference comparisons measured **1.29× faster 2D** (41.28 → 31.90 s) and
**1.66× faster 3D** (102.72 → 61.82 s) on the validated workloads.

GPU support was removed after performance review: measured full-inference
speedups were about 1.33× for large 2D and 1.05× for 3D, while small workloads
were slower. See [the decision and current validation](docs/cpu-only-decision.txt).
Earlier GPU numerical results, rejected float32 experiments and benchmark
reports remain historical evidence in `docs/RESULTS.txt` and related reports;
their GPU commands apply to local revision `39c7eeb`.

## Assumed units

- Time: seconds.
- Length: nanometers.
- Diffusion: nanometers squared per second.
