"""Compare an archived CPU sampler with the optimized sampler, without profiling."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions as f
import model
from scripts.validate_2d import inference, simulate, parameters
from scripts.validate_3d import prepared
from scripts.validate_sampling_defaults import measure


def main(revision: str, directory: Path, output: Path):
    """Benchmark identical workloads and independently validate both targets.

    The archived module is a local measurement reference, never an installed
    inference option. Initialization, model orchestration and GP mathematics
    are unchanged between the two revisions; only the sampler is substituted.
    """
    unchanged = subprocess.check_output([
        "git", "diff", revision, "--", "model.py", "gp_math.py",
        "gp_initialization.py",
    ], text=True)
    if unchanged:
        raise ValueError("Shared model/mathematics changed; use separate checkouts")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gpd_baseline_metropolis.py"
    path.write_text(subprocess.check_output([
        "git", "show", f"{revision}:metropolis.py",
    ], text=True))
    sys.path.insert(0, str(directory.resolve()))
    specification = importlib.util.spec_from_file_location(
        "gpd_baseline_metropolis", path
    )
    baseline = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = baseline
    specification.loader.exec_module(baseline)
    optimized = f.diffusionPointSampler
    optimized_geometry = f.prepare_block_proposals
    records = []
    try:
        for case in ("2d", "3d"):
            for name, sampler, geometry in (
                ("before", baseline.diffusionPointSampler,
                 baseline.prepare_block_proposals),
                ("after", optimized, optimized_geometry),
            ):
                f.diffusionPointSampler = sampler
                f.prepare_block_proposals = geometry
                if case == "2d":
                    data = simulate(1000, 41, True)
                    overrides = {k: parameters()[k] for k in (
                        "nInduX", "nInduY", "nFineX", "nFineY",
                        "covLambda", "covL",
                    )}
                    with contextlib.redirect_stdout(io.StringIO()):
                        model.analyze(
                            data, nIter=1, seed=299, annealing_sweeps=1,
                            adaptation_sweeps=1, burn_in=0, **overrides,
                        )
                    result = inference(100000, "cpu", 41, True)
                    seconds = result["seconds"]
                    assert result["recovery_rms"] < 0.1
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        model.analyze(
                            prepared(count=1000, feature=True), nIter=1,
                            seed=299, annealing_sweeps=1, adaptation_sweeps=1,
                            burn_in=0, nFineX=5, nFineY=5, nFineZ=5,
                            covLambda=0.5, covL=1.5,
                        )
                    result = measure("cpu")
                    seconds = result["full_inference_seconds"]
                record = dict(
                    case=case, variant=name, seconds=seconds, result=result
                )
                records.append(record)
                print(json.dumps(record), flush=True)
    finally:
        f.diffusionPointSampler = optimized
        f.prepare_block_proposals = optimized_geometry
    times = {
        case: {r["variant"]: r["seconds"] for r in records if r["case"] == case}
        for case in ("2d", "3d")
    }
    speedups = {
        case: values["before"] / values["after"]
        for case, values in times.items()
    }
    output.write_text(json.dumps(dict(
        baseline_revision=revision, records=records, speedups=speedups,
        thread_environment={key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
        )},
        qualification=(
            "Sequential isolated warmed before/after measurements without "
            "cProfile; single-workstation observations, not timing confidence intervals."
        ),
    ), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="bb2ddd7")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.baseline, args.directory, args.output)
