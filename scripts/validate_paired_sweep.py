"""Compare actual sweeps using identical valid proposal random inputs."""

import contextlib
import copy
import io
import json
import numpy as np
import cupy as cp

from validate_3d import prepared
import functions as f
import model

data = prepared(count=300)
p = {
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
    base = f.initialization(p, data)
results = []
for seed in (31, 32, 33, 34, 35):
    cpu = copy.deepcopy(base)
    cpu.epsilon = np.full(cpu.nIndu, 0.0001)
    cpu.temperature = 2.0
    gpu = copy.deepcopy(cpu)
    for key, value in vars(cpu).items():
        if isinstance(value, np.ndarray) and key not in (
            "epsilon",
            "delta_t",
            "coordinate_origin",
        ):
            setattr(gpu, key, cp.asarray(value))
    gpu.backend = "gpu"
    f.seed_sampler(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        cpu, accepted_cpu = f.diffusionPointSampler(cpu, data)
    f.seed_sampler(seed)
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        gpu, accepted_gpu = f.diffusionPointSampler(gpu, data)
    np.testing.assert_array_equal(accepted_cpu, accepted_gpu)
    bound = 128 * np.finfo(float).eps * np.linalg.cond(cpu.cProposal)
    np.testing.assert_allclose(
        cpu.dIndu, cp.asnumpy(gpu.dIndu), rtol=bound, atol=bound
    )
    np.testing.assert_allclose(
        cpu.dData, cp.asnumpy(gpu.dData), rtol=bound, atol=bound
    )
    np.testing.assert_allclose(cpu.P, gpu.P, rtol=bound, atol=bound)
    direct = float(
        f.log_likelihood(gpu.dData, gpu.a, cp.asarray(data.delta_t), 3, xp=cp)
        + f.log_prior(gpu.dIndu, gpu.priorMean, gpu.cInduInduInv)
    )
    np.testing.assert_allclose(gpu.P, direct, rtol=bound, atol=bound)
    result = dict(
        seed=seed,
        field_error=float(np.max(abs(cpu.dData - cp.asnumpy(gpu.dData)))),
        posterior_error=abs(cpu.P - gpu.P),
        gpu_posterior_drift=abs(gpu.P - direct),
        accepted=int(accepted_cpu.sum()),
    )
    results.append(result)
    print(result, flush=True)
from pathlib import Path

Path("docs/stage4-paired-sweep-results.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
