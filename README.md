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
is required. The existing inference implementation still uses CPU; installing
CuPy does not establish GPU inference support.

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

Scientific test directories and markers are prepared for Stage 3; no formal
tests exist yet. Once supplied, run hardware suites separately:

```bash
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
GPD_REQUIRE_GPU=1 uv run --no-sync python -m pytest tests/gpu --strict-markers -q
```

In PowerShell set `$env:GPD_REQUIRE_GPU = "1"` before the GPU command and remove
it afterward with `Remove-Item Env:GPD_REQUIRE_GPU`.

See `docs/stage-validation.txt` for executed checks and the incomplete Stage 1 gate.

## Usage

To run the code, simply run the main.py file in your favorite python editor. Change `dataPath` to the path of your data file.

The data file should be a csv file with 3 columns: `particle #`, `xPos`, and `yPos`, where `particle #` is the particle number, `xPos` is the x position of the particle, and `yPos` is the y position of the particle. Position is measured in nanometers. Time steps are assumed to be taken at 30 Hz.

The code will output an image and save it to the `output/` directory.

The code is set to run with default hyperparameters, but the user can change the hyperparameters as needed by including them as keyword arguments in the `analyze` function. A dictionary of the hyperparameters can be found in `model.py`.

## Assumed Units

* Time: Seconds
* Length: Nanometers
