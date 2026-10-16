# GPDiffusionMapping

This is the repository for our Diffusion Mapping project from trajectory data, BioArXiV link. For ease of use, it is organized so that users only need to interact with the main.py file. A synthetic dataset is provided in the /data directory under /example that correlates to a flat Diffusion Map at 0.05 $\frac{\mu m^2}{s}$, where the data set has ~5000 localizations. The experimental data associated with our paper, https://www.biorxiv.org/content/10.1101/2024.02.27.582378v1, is also provided in the /data directory under /experimental.

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

To run the code, simply run the main.py file in your favorite python editor. Change `dataPath` to the path of your data file.

The data file should be a csv file with 3 columns: `particle #`, `xPos`, and `yPos`, where `particle #` is the particle number, `xPos` is the x position of the particle, and `yPos` is the y position of the particle. Position is measured in nanometers. Time steps are assumed to be taken at 30 Hz.

The code will output an image and save it to the `output/` directory.

The code is set to run with default hyperparameters, but the user can change the hyperparameters as needed by including them as keyword arguments in the `analyze` function. A dictionary of the hyperparameters can be found in `model.py`.

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
and `docs/figures/3d-burnin-10000.png`. Stage 4 is incomplete;
the Stage 5 refactoring and CLI have not started. The GPU suite includes this
long-running comparison. See `docs/stage-validation.txt` and
`docs/figures/3d-sampling-1000.png` for evidence.

The computational API accepts `burn_in=10000` to request extra discarded sweeps
after the original annealing and adaptation. Proposal widths remain fixed during
these additional sweeps and production; the default remains zero extra sweeps.

## Assumed Units

* Time: Seconds
* Length: Nanometers
