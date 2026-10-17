"""Profile warmed CPU inference on the scientifically validated workloads."""

import argparse
import contextlib
import cProfile
import io
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_2d import inference
from scripts.validate_sampling_defaults import measure
from scripts.validate_3d import prepared
import model


def main(directory: Path, output: Path, case: str):
    """Save full profiles and compact timings, checking simulated recovery."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name in (("2d", "3d") if case == "both" else (case,)):
        # Warm both annealing and temperature-one Numba signatures before
        # profiling. Shapes are retained so allocation/cache costs are realistic.
        if name == "2d":
            inference(100000, "cpu", 41, True, iterations=5)
            run = lambda: inference(100000, "cpu", 41, True)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                model.analyze(
                    prepared(count=1000, feature=True), nIter=1, seed=299,
                    annealing_sweeps=1, adaptation_sweeps=1, burn_in=0,
                    nFineX=5, nFineY=5, nFineZ=5, covLambda=0.5, covL=1.5,
                )
            run = lambda: measure("cpu")
        profiler = cProfile.Profile()
        result = profiler.runcall(run)
        assert result["recovery_rms" if name == "2d" else "map_rms"] < 0.1
        profiler.dump_stats(directory / f"{name}.prof")
        stats = pstats.Stats(profiler)
        rows = []
        for (filename, line, function), values in stats.stats.items():
            primitive, calls, own, cumulative, _ = values
            rows.append(dict(
                file=filename, line=line, function=function,
                primitive_calls=primitive, calls=calls,
                self_seconds=own, cumulative_seconds=cumulative,
            ))
        record = dict(
            case=name, result=result, profile_seconds=stats.total_tt,
            top_cumulative=sorted(
                rows, key=lambda row: row["cumulative_seconds"], reverse=True
            )[:30],
            top_self=sorted(
                rows, key=lambda row: row["self_seconds"], reverse=True
            )[:20],
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        records.append(record)
        print(json.dumps(dict(case=name, result=result)), flush=True)
    output.write_text(json.dumps(dict(
        python=sys.version, platform=platform.platform(),
        thread_environment={key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
        )},
        records=records,
        qualification="Warmed Linux CPU measurements; profiling adds overhead.",
    ), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("2d", "3d", "both"), default="both")
    args = parser.parse_args()
    main(args.directory, args.output, args.case)
