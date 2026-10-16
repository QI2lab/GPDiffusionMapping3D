"""Independent physical recovery on actual CUDA, with CPU chain comparisons."""

import json
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import t
from scripts.validate_2d import inference, SEEDS

pytestmark=[pytest.mark.gpu,pytest.mark.integration]


@pytest.mark.parametrize('feature',[False,True])
def test_physical_recovery_and_independent_cpu_agreement(cupy_gpu,feature):
    gpu=[inference(10000,'gpu',seed,feature) for seed in SEEDS]
    assert all(r['recovery_rms']<.1 for r in gpu)
    # Independent CPU chains are executed here, not treated as a frozen output.
    cpu=[inference(10000,'cpu',seed,feature) for seed in SEEDS]
    groups=[np.array([r['estimate'] for r in g]) for g in (cpu,gpu)]
    se=np.sqrt(sum(g.var(axis=0,ddof=1)/3 for g in groups))
    bound=t.ppf(1-.01/18,df=2)*se
    assert np.all(abs(groups[0].mean(axis=0)-groups[1].mean(axis=0))<bound)


def test_finer_simulation_preserves_smooth_recovery(cupy_gpu):
    assert inference(10000,'gpu',41,True,substeps=10)['recovery_rms']<.1
