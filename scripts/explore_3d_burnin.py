"""Matched uniform/smooth diagnostics; this experiment does not replace gates."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import io
import json
import multiprocessing
from pathlib import Path

import numpy as np
from scipy import stats

from validate_3d import prepared, recovery, truth
import functions as f
import model


def batch_sensitivity(chains):
    """Report batch-length sensitivity without declaring dependent batches valid."""
    difference = abs(chains[0].mean(axis=0) - chains[1].mean(axis=0))
    results = []
    for size in (100, 250, 500, 1000, 2500):
        count = len(chains[0]) // size
        if count < 2:
            continue
        errors = []
        for chain in chains:
            means = chain[: count * size].reshape(count, size, -1).mean(axis=1)
            errors.append(means.std(axis=0, ddof=1) / np.sqrt(count))
        bound = stats.t.ppf(1 - 0.01 / (2 * 27), count - 1) * np.sqrt(
            errors[0] ** 2 + errors[1] ** 2
        )
        results.append(
            {
                "batch_length": size,
                "batch_count": count,
                "max_combined_mcse": float(np.max(np.hypot(*errors))),
                "center_combined_mcse": float(np.hypot(*errors)[13]),
                "positions_outside_nominal_bound": int(
                    np.sum(difference >= bound)
                ),
                "qualification": "Exploratory; independence and CLT not established",
            }
        )
    return results


def split_rhat(chains):
    """Calculate rank-normalized and folded split R-hat for matched chains."""
    half = len(chains[0]) // 2
    split = np.stack(
        [part for chain in chains for part in (chain[:half], chain[-half:])]
    )

    def normalized_rhat(values):
        ranks = stats.rankdata(values.reshape(-1, values.shape[-1]), axis=0)
        z = stats.norm.ppf((ranks - 0.375) / (len(ranks) + 0.25))
        z = z.reshape(values.shape)
        within = z.var(axis=1, ddof=1).mean(axis=0)
        between = half * z.mean(axis=1).var(axis=0, ddof=1)
        return np.sqrt(((half - 1) / half * within + between / half) / within)

    rank = normalized_rhat(split)
    folded = normalized_rhat(abs(split - np.median(split, axis=(0, 1))))
    return {
        "maximum_rank_split_rhat": float(rank.max()),
        "maximum_folded_split_rhat": float(folded.max()),
        "positions_above_1_01": int(np.sum(np.maximum(rank, folded) > 1.01)),
        "qualification": "Two independent chains only; diagnostic, not proof",
    }


def geometry(feature):
    """Assess grid representation and local likelihood information independently."""
    data = prepared(seed=41, feature=feature)
    parameters = {
        **model.PARAMETERS,
        "nInduX": 5,
        "nInduY": 5,
        "nInduZ": 5,
        "nFineX": 5,
        "nFineY": 5,
        "nFineZ": 5,
        "covLambda": 0.5,
        "covL": 1.5,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        v = f.initialization(parameters, data)
    weights = v.cDataIndu
    target = truth(data.starts, feature)
    fit, _, _, singular = np.linalg.lstsq(weights, target, rcond=None)
    predicted = weights @ fit
    points = np.array(
        [
            (z, y, x)
            for z in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            for x in (-1.0, 0.0, 1.0)
        ]
    )
    central = f.covMat(points, v.induCoordinates, 0.5, 1.5) @ v.cInduInduInv
    fisher = weights.T @ ((1.5 / target**2)[:, None] * weights)
    spectrum = np.linalg.eigvalsh(fisher)
    inducing_precision = fisher + v.cInduInduInv
    alpha_precision = v.cProposal @ inducing_precision @ v.cProposal
    chol = np.linalg.cholesky(v.cProposal)
    whitened_precision = chol.T @ inducing_precision @ chol
    alpha_scale = np.sqrt(np.diag(alpha_precision))
    squared = np.sum((data.ends - data.starts) ** 2, axis=1)
    mle = float(np.mean(squared / (6 * data.delta_t)))
    n = len(squared)
    interval = [
        float(3 * n * mle / stats.chi2.ppf(q, 3 * n)) for q in (0.995, 0.005)
    ]
    return {
        "data_seed": 41,
        "transitions": n,
        "inducing_points": v.nIndu,
        "gp_covariance_condition": float(np.linalg.cond(v.cProposal)),
        "design_condition": float(singular[0] / singular[-1]),
        "expected_likelihood_fisher_condition": float(
            spectrum[-1] / spectrum[0]
        ),
        "expected_posterior_precision_condition_inducing": float(
            np.linalg.cond(inducing_precision)
        ),
        "expected_posterior_precision_condition_alpha": float(
            np.linalg.cond(alpha_precision)
        ),
        "expected_alpha_precision_condition_after_diagonal_scaling": float(
            np.linalg.cond(
                alpha_precision / np.outer(alpha_scale, alpha_scale)
            )
        ),
        "expected_posterior_precision_condition_whitened": float(
            np.linalg.cond(whitened_precision)
        ),
        "inducing_grid_minimum": v.induCoordinates.min(axis=0).tolist(),
        "inducing_grid_maximum": v.induCoordinates.max(axis=0).tolist(),
        "unconstrained_truth_fit_rms_observations": float(
            np.sqrt(np.mean(((predicted - target) / target) ** 2))
        ),
        "unconstrained_truth_fit_rms_central": float(
            np.sqrt(
                np.mean(
                    (
                        (central @ fit - truth(points, feature))
                        / truth(points, feature)
                    )
                    ** 2
                )
            )
        ),
        "fit_minimum_inducing": float(fit.min()),
        "fit_minimum_observation": float(predicted.min()),
        "minimum_observations_within_radius_0_5_of_central_points": int(
            min(
                np.sum(np.linalg.norm(data.starts - point, axis=1) < 0.5)
                for point in points
            )
        ),
        "scalar_constant_mle": mle,
        "scalar_constant_99_percent_chi_square_interval": interval
        if not feature
        else None,
        "scalar_control_qualification": "Exact for uniform independent increments only; not GP posterior",
    }


def run_chain(backend, feature, directory, burn_in):
    """Retain the complete production trace and sampler diagnostics locally."""
    directory = Path(directory)
    seed = 41 if backend == "cpu" else 42
    with (directory / f"{backend}.log").open("w", buffering=1) as stream:
        with contextlib.redirect_stdout(stream):
            result = recovery(
                backend,
                seed,
                feature,
                iterations=5000,
                data_seed=41,
                burn_in=burn_in,
                quiet=False,
                trace_path=directory / f"{backend}.npz",
            )
    (directory / f"{backend}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return feature, backend, result


def summarize(directory, feature=False):
    """Summarize actual traces without weakening the existing acceptance rule."""
    chains = []
    diagnostics = {}
    data = prepared(seed=41, feature=feature)
    axes = [
        np.linspace(low, high, 5)
        for low, high in zip(
            data.trajectories.min(axis=0), data.trajectories.max(axis=0)
        )
    ]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    kernel = f.covMat(coordinates, coordinates, 0.5, 1.5)
    covariance = kernel + 0.01 * kernel.mean() * np.eye(125)
    inverse = np.linalg.inv(covariance)
    basis = f.covMat(data.starts, coordinates, 0.5, 1.5)
    weights = basis @ inverse
    expected_conditional_alpha_sd = 1 / np.sqrt(
        (1.5 * (basis / truth(data.starts, feature)[:, None]) ** 2).sum(axis=0)
        + np.diag(covariance)
    )
    steps = data.ends - data.starts
    prior_mean = np.mean(np.sum(steps**2, axis=1) / (6 * data.delta_t))
    for backend in ("cpu", "gpu"):
        with np.load(directory / f"{backend}.npz") as trace:
            chains.append(trace["projected"].copy())
            epsilon = trace["epsilon"]
            acceptance = trace["acceptance_fraction"]
            ranges = np.ptp(trace["samples"], axis=0)
            alpha = trace["samples"] @ inverse.T
            alpha_ranges = np.ptp(alpha, axis=0)
            roundoff = (
                64
                * np.finfo(float).eps
                * np.linalg.cond(covariance)
                * max(1.0, float(abs(alpha).max()))
            )
            posterior_errors = []
            for index in (
                0,
                len(trace["samples"]) // 2,
                len(trace["samples"]) - 1,
            ):
                inducing = trace["samples"][index]
                field = weights @ inducing
                log_density = stats.norm.logpdf(
                    steps,
                    scale=np.sqrt(2 * field[:, None] * data.delta_t[:, None]),
                ).sum()
                residual = inducing - prior_mean
                log_density -= 0.5 * residual @ inverse @ residual
                posterior_errors.append(
                    float(abs(log_density - trace["posterior"][index]))
                )
            diagnostics[backend] = {
                "epsilon_min": float(epsilon.min()),
                "epsilon_median": float(np.median(epsilon)),
                "epsilon_max": float(epsilon.max()),
                "nonfinite_epsilon": int(np.sum(~np.isfinite(epsilon))),
                "zero_epsilon": int(np.sum(epsilon == 0)),
                "minimum_acceptance": float(acceptance.min()),
                "median_acceptance": float(np.median(acceptance)),
                "maximum_acceptance": float(acceptance.max()),
                "zero_acceptance_coefficients": int(np.sum(acceptance == 0)),
                "minimum_inducing_production_range": float(ranges.min()),
                "minimum_absolute_net_alpha_change": float(
                    abs(trace["final_alpha"] - trace["initial_alpha"]).min()
                ),
                "minimum_alpha_production_range": float(alpha_ranges.min()),
                "median_alpha_production_range": float(
                    np.median(alpha_ranges)
                ),
                "alpha_roundoff_resolution_bound": roundoff,
                "alpha_ranges_below_roundoff_bound": int(
                    np.sum(alpha_ranges <= roundoff)
                ),
                "maximum_saved_posterior_error_three_independent_gaussian_checks": max(
                    posterior_errors
                ),
                "zero_acceptance_details": {
                    "indices": np.flatnonzero(acceptance == 0).tolist(),
                    "epsilon": epsilon[acceptance == 0].tolist(),
                    "initial_alpha": trace["initial_alpha"][
                        acceptance == 0
                    ].tolist(),
                    "mean_proposal_standard_deviation": (
                        epsilon * abs(trace["initial_alpha"])
                    )[acceptance == 0].tolist(),
                    "expected_local_conditional_standard_deviation": expected_conditional_alpha_sd[
                        acceptance == 0
                    ].tolist(),
                    "qualification": "Local Fisher approximation at simulation truth; not exact conditional quantiles",
                },
            }
        diagnostics[backend]["chain"] = json.loads(
            (directory / f"{backend}.json").read_text()
        )
    return {
        "chains": diagnostics,
        "rhat": split_rhat(chains),
        "batch_sensitivity": batch_sensitivity(chains),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        results = json.loads(args.output.read_text())
        for feature in (False, True):
            name = "smooth" if feature else "uniform"
            results["cases"][name] = {
                "geometry": geometry(feature),
                **summarize(args.directory / name, feature),
            }
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Updated {args.output}", flush=True)
        return
    results = {
        "protocol": {
            "data_seed": 41,
            "cpu_seed": 41,
            "gpu_seed": 42,
            "annealing_sweeps": 10000,
            "adaptation_sweeps": 200,
            "additional_fixed_width_burn_in": args.burn_in,
            "production_sweeps": 5000,
        },
        "cases": {},
    }
    for feature in (False, True):
        name = "smooth" if feature else "uniform"
        directory = args.directory / name
        directory.mkdir(parents=True, exist_ok=True)
        results["cases"][name] = {"geometry": geometry(feature)}
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_chain,
                backend,
                feature,
                args.directory / ("smooth" if feature else "uniform"),
                args.burn_in,
            )
            for feature in (False, True)
            for backend in ("cpu", "gpu")
        ]
        for future in as_completed(futures):
            feature, backend, result = future.result()
            print(
                json.dumps(
                    {
                        "case": "smooth" if feature else "uniform",
                        "backend": backend,
                        "seconds": result["seconds"],
                        "sampling": result["sampling"],
                    }
                ),
                flush=True,
            )
    for name in results["cases"]:
        results["cases"][name].update(
            summarize(args.directory / name, name == "smooth")
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
