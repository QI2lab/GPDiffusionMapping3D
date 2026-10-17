# GPDiffusionMapping

Gaussian-process diffusion mapping from already linked particle trajectories.
The model infers a static isotropic diffusion field in XY or ZYX coordinates.
Synthetic and experimental data accompanying the
[paper](https://www.biorxiv.org/content/10.1101/2024.02.27.582378v1) are in `data/`.

## Installation

### Download

To download the package, simply clone this repository using `git` or click the green <>Code button above and select the download option. This will download the package as a Zip file to your computer. Unzip the file and open the directory in your favorite python editor.

#### Setup environment

Use Python 3.12 and uv from the checkout directory. `pyproject.toml` is the
dependency specification. Resolve a local environment with:

```bash
uv sync
uv sync --group dev --extra gpu
```

Windows/Linux select the project's GPU extra through the default `gpu` group.
This installs `cupy-cuda12x[ctk]` and its CUDA component wheels; an NVIDIA driver
is required. The existing 2D inference now supports explicit CPU/GPU float64
execution. Installation alone does not establish scientific GPU validation;
executed results and operating conditions are recorded in `docs/stage-validation.txt`.

For a CPU-only environment:

```bash
uv sync --group dev --no-default-groups
```

Use `uv run --no-sync` for subsequent commands so the chosen environment stays
selected. `uv.lock` is local and ignored; fresh checkouts resolve with `uv sync`.

The standalone migration validation is:

```bash
uv run --no-sync python scripts/validate_migration.py
```

Mathematics and physics tests are separated by hardware. Run the suites separately:

```bash
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
GPD_REQUIRE_GPU=1 uv run --no-sync python -m pytest tests/gpu --strict-markers -q
```

In PowerShell set `$env:GPD_REQUIRE_GPU = "1"` before the GPU command and remove
it afterward with `Remove-Item Env:GPD_REQUIRE_GPU`.

See `docs/stage-validation.txt` for executed checks and stage status.

## Usage

The `gpdiffusion-map` Typer command reads `.csv` or `.parquet`. Input contains
already linked tracks; timestamps are numeric seconds. Positions default to
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
uv run --no-sync gpdiffusion-map tracks.csv --backend gpu
uv run --no-sync gpdiffusion-map tracks.parquet --backend cpu
uv run --no-sync gpdiffusion-map tracks.parquet --particle-col track_id --z-col z_um --y-col y_um --x-col x_um --time-col time_s --scale 1000
uv run --no-sync gpdiffusion-map legacy.csv --dimensions 2 --x-col xPos --y-col yPos --delta-t 0.03333333333333333
uv run --no-sync python -m gpdiffusion_cli tracks.csv --backend gpu
uv run --no-sync gpdiffusion-map --help
```

The scale example converts micrometers to nanometers. Column overrides are
exact and apply to both formats. Unrelated columns are ignored and particle
identities are preserved without conversion to floating point. Three-dimensional
input always requires timestamps. Legacy XY input without a timestamp column
requires an explicit positive `--delta-t`; it uses row order within each linked
particle. `--delta-t` is rejected when timestamps are present.

Defaults are `--dimensions 3`, `--backend gpu`, `--dtype float64`,
`--iterations 1000`, `--output-dir output` and `--scale 1`.
GPU float32 failed validation and is rejected by the file/CLI inference path.
An unavailable requested GPU is an error; inference never switches to CPU.

Outputs are `InferredMAP.png`, `inference.npz`, `metadata.json` and
`sampling.log`. The figure uses µm and µm²/s; numerical coordinates and diffusion
samples use nm and nm²/s. The 3D figure is an XY slice at the nearest central Z.
NPZ contains every production inducing sample, unnormalized log posterior,
inducing/fine coordinates and the MAP field on the fine grid. Metadata records
the coordinate order, units, backend and sampling schedule.

File reading, inference and output are callable separately through
`data_reader.read_tracks`, `model.analyze`, `mapping_io.infer_file` and
`mapping_io.save_results`. Model parameter overrides remain available through
the computational functions; see `model.PARAMETERS`. `main.py` also invokes
the same CLI from a checkout.

## Validated precision and stage status

2D float64 CPU/GPU inference has passed the simulation and mathematical gates.
Select GPU explicitly through the computational API:

```python
import data_reader
import model

data = data_reader.data_reader("data/synthetic/simulated_flat_Dmap.csv")
variables, samples, posterior = model.analyze(data, backend="gpu", seed=41)
```

CPU remains available with `backend="cpu"` and does not import CuPy. GPU float32
was rejected for inaccurate proposal likelihood ratios and remains a diagnostic
option with an explicit warning; use float64 for inference.

Timestamped 3D numerical and physical-recovery cases pass, but independent
chain-mean agreement failed at 1,000 and 5,000 production iterations, with slow
mixing and batch-length-sensitive uncertainty estimates. A matched uniform/smooth
investigation added 10,000 discarded temperature-one sweeps after adaptation,
before 5,000 production sweeps. Both fields still mixed poorly: maximum split
R-hat 1.78 (uniform) / 1.87 (smooth), with coefficients that never updated in
production. See `docs/3d-burnin-investigation.txt`, `docs/stage4-burnin-10000.json`
and `docs/figures/3d-burnin-10000.png` for this original-sampler investigation.

The authorized repair adds fixed Fisher-preconditioned block Metropolis
proposals and recent-window warmup tuning, preserving the exact likelihood,
GP prior and positivity constraints. Eight-coordinate blocks passed the
development checks on both CPU and CUDA: maximum rank/folded R-hat 1.00536,
minimum bulk ESS 2,127 across all 278 monitored quantities, and physical RMS
recovery below 6.84%. Held-out confirmation also passed: maximum R-hat 1.00501,
minimum bulk ESS 2,083 and recovery below 6.33%. The full required-GPU suite
passed 30 tests with zero skips; the separate CPU suite passed 40 tests.
Stage 4 is complete. Stage 5 refactoring and CLI are implemented; their complete
CPU/GPU scientific suites and final performance measurements are in progress.
See `docs/sampling-repair-plan.txt`,
`docs/sampling-blocked8-development.json` and
`docs/figures/sampling-blocked8-development.png`. The full GPU suite includes
four-chain convergence checks for both uniform and smooth fields.

For 3D, the computational API now defaults to eight-coordinate blocked proposals,
2,000 annealing sweeps, 4,000 temperature-one adaptation sweeps and 1,000
additional discarded sweeps. Geometry and widths stay fixed in production.
Use `proposal="relative", adaptation="legacy"` for the historical 3D reference.
2D retains its original defaults. `burn_in` overrides additional discarded
sweeps. Production defaults to 1,000 as before; the convergence validation used
four independent chains with 20,000 production sweeps each. A short single
run does not establish posterior convergence.
Use explicit independent seeds and varied positive initial fields through the
computational API for multiple-chain diagnostics; the validation protocol is in
`scripts/run_sampling_repair.py`. Repeating a CLI command is not a substitute
for that independent-chain protocol.

## Assumed Units

* Time: Seconds
* Length: Nanometers
