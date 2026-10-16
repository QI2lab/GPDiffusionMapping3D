"""Execute the predeclared 2D gate sequentially on local CPU and CUDA."""

import json
from pathlib import Path
import numpy as np
from scipy.stats import t

from validate_2d import mathematical_checks, inference, benchmark, SEEDS

results = []
path = Path("docs/stage1-results.json")


def record(kind, result):
    result = dict(kind=kind, **result)
    results.append(result)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(result), flush=True)


record("math", mathematical_checks())
for count in (1000, 100000):
    for backend in ("cpu", "gpu"):
        record("benchmark", benchmark(count, backend))
for feature in (False, True):
    for seed in SEEDS:
        for backend in ("cpu", "gpu"):
            result = inference(10000, backend, seed, feature)
            record("recovery", result)
            assert result["recovery_rms"] < 0.10, result
for backend in ("cpu", "gpu"):
    result = inference(10000, backend, 41, True, substeps=10)
    record("fine_simulator", result)
    assert result["recovery_rms"] < 0.10, result
for backend in ("cpu", "gpu"):
    record("end_to_end", inference(100000, backend, 41, True))
for feature in (False, True):
    groups = [
        np.array(
            [
                r["estimate"]
                for r in results
                if r["kind"] == "recovery"
                and r["feature"] == feature
                and r["backend"] == backend
            ]
        )
        for backend in ("cpu", "gpu")
    ]
    difference = np.abs(groups[0].mean(axis=0) - groups[1].mean(axis=0))
    se = np.sqrt(sum(g.var(axis=0, ddof=1) / len(g) for g in groups))
    bound = t.ppf(1 - 0.01 / (2 * 9), df=2) * se
    assert np.all(difference < bound), (difference, bound)
    record(
        "chain_agreement",
        dict(
            feature=feature,
            max_difference=float(difference.max()),
            minimum_bound=float(bound.min()),
        ),
    )
print("STAGE 1 SCIENTIFIC CHECKS PASSED", flush=True)
