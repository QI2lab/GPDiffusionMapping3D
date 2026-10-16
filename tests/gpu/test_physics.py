"""Independent physical recovery on actual CUDA, with CPU chain comparisons."""

import json
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import t
from scripts.validate_2d import inference, SEEDS

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("feature", [False, True])
def test_physical_recovery_and_independent_cpu_agreement(cupy_gpu, feature):
    from scripts.validate_2d import independent_backend_checks

    independent_backend_checks(feature)


def test_finer_simulation_preserves_smooth_recovery(cupy_gpu):
    assert inference(10000, "gpu", 41, True, substeps=10)["recovery_rms"] < 0.1
