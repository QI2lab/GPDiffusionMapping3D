"""Predeclared local sampling-repair experiments on the original 3D fields."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_3d import prepared, truth
from scripts.sampling_statistics import diagnose, passes
import functions as f
import model


def run_chain(
    directory, backend, feature, index, proposal, pilot, heldout, block_size=16
):
    """Run one independent chain, preserving logs, all draws and warmup history."""
    directory = Path(directory)
    seed = (
        (301 if backend == "cpu" else 401) + index + (1000 if heldout else 0)
    )
    data_seed = (43 if feature else 42) if heldout else 41
    data = prepared(
        seed=data_seed,
        feature=feature,
        irregular=not (heldout and not feature),
        substeps=10 if heldout and feature else 1,
    )
    parameters = {
        "nInduX": 5,
        "nInduY": 5,
        "nInduZ": 5,
        "nFineX": 5,
        "nFineY": 5,
        "nFineZ": 5,
        "covLambda": 0.5,
        "covL": 1.5,
    }
    if backend == "gpu":
        import cupy as cp

        cp.cuda.Device(index % min(2, cp.cuda.runtime.getDeviceCount())).use()
        probe = (cp.arange(32, dtype=cp.float64) ** 2).sum()
        cp.cuda.Device().synchronize()
        assert float(probe) == 10416
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)[
            "name"
        ].decode()
    else:
        device = "CPU"
    with contextlib.redirect_stdout(io.StringIO()):
        initial = f.initialization({**model.PARAMETERS, **parameters}, data)
    rng = np.random.default_rng(seed + 10000)
    factor = (0.7, 0.9, 1.1, 1.3)[index]
    field = (
        factor * initial.priorMean
        + 0.2 * initial.cInduInduChol @ rng.normal(size=125)
    )
    if np.any(field <= 0) or np.any(initial.cDataIndu @ field <= 0):
        raise ValueError(
            "Predeclared initial field is invalid; do not clip it"
        )
    prefix = directory / f"{backend}-{index}"
    started = time.perf_counter()
    with prefix.with_suffix(".log").open("w", buffering=1) as stream:
        with contextlib.redirect_stdout(stream):
            v, draws, posterior = model.analyze(
                data,
                nIter=1000 if pilot else 20000,
                backend=backend,
                seed=seed,
                proposal=proposal,
                block_size=block_size,
                adaptation="diminishing",
                annealing_sweeps=500 if pilot else 2000,
                adaptation_sweeps=4000,
                burn_in=1000,
                diagnostics=True,
                initial_inducing=field,
                **parameters,
            )
    points = np.array(
        [
            (z, y, x)
            for z in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            for x in (-1.0, 0.0, 1.0)
        ]
    )
    weights = f.covMat(points, v.induCoordinates, 0.5, 1.5) @ v.cInduInduInv
    projected = draws @ weights.T
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        samples=draws,
        alpha=draws @ v.cInduInduInv.T,
        projected=projected,
        posterior=posterior,
        weights=weights,
        inverse=v.cInduInduInv,
        coordinates=v.induCoordinates,
        **v.sampling_metadata,
    )
    np.savez_compressed(
        prefix.with_suffix(".warmup.npz"),
        phase=np.array([row[0] for row in v.warmup_trace]),
        sweep=np.array([row[1] for row in v.warmup_trace]),
        widths=np.array([row[2] for row in v.warmup_trace]),
        acceptance=np.array([row[3] for row in v.warmup_trace]),
        proposal_stats=np.array([row[4] for row in v.warmup_trace]),
    )
    result = {
        "backend": backend,
        "seed": seed,
        "data_seed": data_seed,
        "feature": feature,
        "proposal": proposal,
        "block_size": block_size,
        "device": device,
        "seconds": time.perf_counter() - started,
        "map_rms": float(
            np.sqrt(
                np.mean(
                    (
                        (
                            projected[np.argmax(posterior)]
                            - truth(points, feature)
                        )
                        / truth(points, feature)
                    )
                    ** 2
                )
            )
        ),
        "acceptance_min": float(
            v.sampling_metadata["acceptance_fraction"].min()
        ),
        "acceptance_max": float(
            v.sampling_metadata["acceptance_fraction"].max()
        ),
        "zero_acceptance": int(
            sum(v.sampling_metadata["acceptance_fraction"] == 0)
        ),
        "width_min": float(v.sampling_metadata["epsilon"].min()),
        "width_max": float(v.sampling_metadata["epsilon"].max()),
        "gpu_pool_bytes": cp.get_default_memory_pool().total_bytes()
        if backend == "gpu"
        else 0,
    }
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize(directory, count, pilot):
    """Apply unchanged scientific criteria to all field/coefficient quantities."""
    results = {}
    observation_weights = None
    for backend in ("cpu", "gpu"):
        chains = []
        individuals = []
        for index in range(count):
            prefix = directory / f"{backend}-{index}"
            with np.load(prefix.with_suffix(".npz")) as trace:
                chains.append(
                    np.concatenate(
                        (
                            trace["samples"],
                            trace["alpha"],
                            trace["projected"],
                            trace["posterior"][:, None],
                        ),
                        axis=1,
                    )
                )
                individual = json.loads(
                    prefix.with_suffix(".json").read_text()
                )
                if observation_weights is None:
                    data = prepared(
                        seed=individual["data_seed"],
                        feature=individual["feature"],
                        irregular=not (
                            individual["data_seed"] == 42
                            and not individual["feature"]
                        ),
                        substeps=10
                        if individual["data_seed"] == 43
                        and individual["feature"]
                        else 1,
                    )
                    coordinates = trace["coordinates"]
                    kernel = 0.25 * np.exp(
                        -np.sum(
                            (coordinates[:, None] - coordinates[None, :]) ** 2,
                            axis=2,
                        )
                        / 4.5
                    )
                    covariance = kernel + 0.01 * kernel.mean() * np.eye(125)
                    cross = 0.25 * np.exp(
                        -np.sum(
                            (data.starts[:, None] - coordinates[None, :]) ** 2,
                            axis=2,
                        )
                        / 4.5
                    )
                    observation_weights = np.linalg.solve(
                        covariance, cross.T
                    ).T
                    prior_mean = np.mean(
                        np.sum((data.ends - data.starts) ** 2, axis=1)
                        / (6 * data.delta_t)
                    )
                maximum_error = 0.0
                minimum_bound = np.inf
                for point in (
                    0,
                    len(trace["samples"]) // 2,
                    len(trace["samples"]) - 1,
                ):
                    inducing = trace["samples"][point]
                    field = observation_weights @ inducing
                    if (
                        np.any(inducing <= 0)
                        or np.any(field <= 0)
                        or not np.all(np.isfinite(field))
                    ):
                        maximum_error = np.inf
                        continue
                    residual = inducing - prior_mean
                    direct = stats.norm.logpdf(
                        data.ends - data.starts,
                        scale=np.sqrt(
                            2 * field[:, None] * data.delta_t[:, None]
                        ),
                    ).sum()
                    direct -= (
                        0.5 * residual @ np.linalg.solve(covariance, residual)
                    )
                    if not np.isfinite(direct):
                        maximum_error = np.inf
                        continue
                    maximum_error = max(
                        maximum_error,
                        float(abs(direct - trace["posterior"][point])),
                    )
                    minimum_bound = min(
                        minimum_bound,
                        128
                        * np.finfo(float).eps
                        * np.linalg.cond(covariance)
                        * max(1, abs(direct)),
                    )
                individual["posterior_consistency_max_error"] = maximum_error
                individual["posterior_consistency_bound"] = minimum_bound
                individual["posterior_consistency_pass"] = bool(
                    maximum_error < minimum_bound
                )
            individuals.append(individual)
        chains = np.stack(chains)
        checkpoints = [1000] if pilot else [5000, 10000, 20000]
        measurements = []
        for size in checkpoints:
            d = diagnose(chains[:, :size])
            measurements.append(
                {
                    "production_sweeps": size,
                    "max_rhat": float(d["rhat"].max()),
                    "min_bulk_ess": float(d["bulk_ess"].min()),
                    "min_tail_ess": float(d["tail_ess"].min()),
                    "max_relative_mcse": float(
                        (d["mean_mcse"] / d["posterior_sd"]).max()
                    ),
                    "convergence_pass": passes(d),
                    "diagnostics": {k: v.tolist() for k, v in d.items()},
                }
            )
        results[backend] = {
            "chains": individuals,
            "checkpoints": measurements,
            "scientific_pass": bool(
                not pilot
                and measurements[-1]["convergence_pass"]
                and all(
                    r["map_rms"] < 0.1
                    and r["zero_acceptance"] == 0
                    and r["posterior_consistency_pass"]
                    for r in individuals
                )
            ),
        }
    # Agreement uses means from independently initialized chains, not paired draws.
    groups = []
    errors = []
    for backend in ("cpu", "gpu"):
        arrays = np.stack(
            [
                np.load(directory / f"{backend}-{i}.npz")["projected"]
                for i in range(count)
            ]
        )
        groups.append(arrays.mean(axis=(0, 1)))
        errors.append(diagnose(arrays)["mean_mcse"])
    difference = abs(groups[0] - groups[1])
    bound = stats.t.ppf(1 - 0.01 / (2 * 27), df=count - 1) * np.hypot(*errors)
    results["backend_agreement"] = {
        "difference": difference.tolist(),
        "bound": bound.tolist(),
        "nominal_pass": bool(np.all(difference < bound)),
        "qualification": "Only meaningful after both backends converge",
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposal", choices=("additive", "blocked"), required=True
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    count = 2 if args.pilot else 4
    if not args.summarize_only:
        tasks = []
        for feature in (False, True):
            directory = args.directory / ("smooth" if feature else "uniform")
            directory.mkdir(parents=True, exist_ok=True)
            tasks.extend(
                (
                    directory,
                    backend,
                    feature,
                    index,
                    args.proposal,
                    args.pilot,
                    args.heldout,
                    args.block_size,
                )
                for backend in ("cpu", "gpu")
                for index in range(count)
            )
        with ProcessPoolExecutor(
            max_workers=8, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for future in as_completed(
                [pool.submit(run_chain, *task) for task in tasks]
            ):
                print(json.dumps(future.result()), flush=True)
    print("Analyzing independent chains for both physical fields", flush=True)
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as analysis:
        summaries = {
            name: analysis.submit(
                summarize, args.directory / name, count, args.pilot
            )
            for name in ("uniform", "smooth")
        }
        cases = {name: future.result() for name, future in summaries.items()}
    results = {
        "proposal": args.proposal,
        "block_size": args.block_size,
        "pilot": args.pilot,
        "heldout": args.heldout,
        "chains_per_backend": count,
        "checkpoints": [1000] if args.pilot else [5000, 10000, 20000],
        "cases": cases,
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)
    if not args.pilot and not all(
        case["cpu"]["scientific_pass"]
        and case["gpu"]["scientific_pass"]
        and case["backend_agreement"]["nominal_pass"]
        for case in results["cases"].values()
    ):
        raise SystemExit(
            "Scientific sampling gate failed; inspect the saved diagnostics"
        )


if __name__ == "__main__":
    main()
